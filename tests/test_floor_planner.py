from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from stc.sched import (
    CpuModel,
    FloorOp,
    FloorPlannerError,
    FloorProgram,
    InstructionForm,
    RegisterFileSpec,
    ResourceSpec,
    cpu_from_agner_csv,
    circuit_to_floor_program,
    emit_circuit_x86_64_asm,
    emit_x86_64_asm,
    plan_floor,
)
from stc.circuit_synth import CircuitState
from stc.mir import Binary, MIRFunction, VReg
from stc.mir.lower_floor import emit_mir_x86_64_asm, plan_mir_floor


TABLE = (
    Path(__file__).resolve().parents[1]
    / "inference"
    / "results"
    / "agner_zen5_avx512.csv"
)


def _cpu(*, register_count: int = 32):
    return cpu_from_agner_csv(
        TABLE,
        name="zen5-floor-test",
        issue_width=4,
        resources=tuple(ResourceSpec(f"P{i}") for i in range(4)),
        register_files=(RegisterFileSpec("zmm", register_count, 512),),
        features=("avx512f", "avx512vnni", "avx512bf16", "fma3"),
    )


class TestFloorPlanner(unittest.TestCase):
    def test_circuit_state_bitwise_subset_reaches_direct_assembly(self):
        circuit = CircuitState(
            input_bits=4,
            output_bits=1,
            gates=[("xor", 0, 1), ("and", 2, 3), ("or", 4, 5)],
            outputs=[(6, False)],
            gate_count=3,
        )

        program = circuit_to_floor_program(circuit)
        assembly = emit_circuit_x86_64_asm(circuit, _cpu(), function_name="circuit_floor_v1")

        self.assertEqual(program.outputs, (6,))
        self.assertIn("circuit_floor_v1:", assembly)
        self.assertIn("vpxorq", assembly)

    def test_mir_bitwise_subset_has_a_gcc_free_path(self):
        mir = MIRFunction(
            input_regs=[VReg(0), VReg(1), VReg(2), VReg(3)],
            output_regs=[VReg(6)],
            instructions=[
                Binary(dst=VReg(4), op="xor", a=VReg(0), b=VReg(1)),
                Binary(dst=VReg(5), op="and", a=VReg(2), b=VReg(3)),
                Binary(dst=VReg(6), op="or", a=VReg(4), b=VReg(5)),
            ],
            num_virtual_regs=7,
        )

        program, schedule = plan_mir_floor(mir, _cpu())
        assembly = emit_mir_x86_64_asm(mir, _cpu(), function_name="mir_floor_v1")

        self.assertEqual(program.outputs, (6,))
        self.assertEqual(schedule.placement(2).cycle, 2)
        self.assertIn("mir_floor_v1:", assembly)
        self.assertNotIn("_mm512", assembly)

    def test_shared_ports_are_reserved_across_instruction_families(self):
        # VNNI and FMA are different forms, but both are limited to P0/P1 at
        # two instructions per cycle.  Four issue slots must not make the
        # shared execution pair disappear.
        operations = []
        outputs = []
        input_values = set()
        value = 12
        for op_id in range(6):
            family = "vnni8" if op_id % 2 == 0 else "fma"
            inputs = (value - 3, value - 2, value - 1)
            input_values.update(inputs)
            operations.append(
                FloorOp(
                    id=op_id,
                    family=family,
                    operands="v,v,v",
                    inputs=inputs,
                    output=value,
                )
            )
            outputs.append(value)
            value += 4
        program = FloorProgram(
            inputs=tuple(sorted(input_values)),
            outputs=tuple(outputs),
            operations=tuple(operations),
        )

        schedule = plan_floor(program, _cpu())

        self.assertEqual(schedule.total_cycles, 3)
        self.assertEqual(
            [len(schedule.operations_at(cycle)) for cycle in range(3)], [2, 2, 2]
        )
        for cycle in range(3):
            self.assertEqual(schedule.resource_usage(cycle), {"P0": 1, "P1": 1})

    def test_port_matching_preserves_narrow_resource_slots(self):
        cpu = CpuModel(
            name="matching-test",
            issue_width=2,
            resources=(ResourceSpec("P0", 1), ResourceSpec("P1", 1)),
            register_files=(RegisterFileSpec("zmm", 8, 512),),
            instruction_forms=(
                InstructionForm(
                    mnemonic="NARROW",
                    operands="v,v",
                    uops=1,
                    latency=1,
                    reciprocal_throughput=1,
                    pipes=("P0",),
                    family="narrow",
                ),
                InstructionForm(
                    mnemonic="WIDE",
                    operands="v,v",
                    uops=1,
                    latency=1,
                    reciprocal_throughput=1,
                    pipes=("P0", "P1"),
                    family="wide",
                ),
            ),
        )
        program = FloorProgram(
            inputs=(0, 1, 2, 3),
            outputs=(4, 5),
            operations=(
                FloorOp(0, "wide", (0, 1), 4, operands="v,v"),
                FloorOp(1, "narrow", (2, 3), 5, operands="v,v"),
            ),
        )

        schedule = plan_floor(program, cpu)

        self.assertEqual(schedule.total_cycles, 1)
        self.assertEqual(schedule.placement(0).resources, ("P1",))
        self.assertEqual(schedule.placement(1).resources, ("P0",))

    def test_dependency_latency_and_non_topological_input_order(self):
        # The operation list is intentionally reversed.  The dependency walk,
        # rather than source order, must derive the critical path.
        first = FloorOp(
            id=10,
            family="vnni8",
            operands="v,v,v",
            inputs=(0, 1, 2),
            output=3,
        )
        second = FloorOp(
            id=11,
            family="vnni8",
            operands="v,v,v",
            inputs=(3, 4, 5),
            output=6,
        )
        schedule = plan_floor(
            FloorProgram(
                inputs=(0, 1, 2, 4, 5),
                outputs=(6,),
                operations=(second, first),
            ),
            _cpu(),
        )

        self.assertEqual(schedule.placement(10).cycle, 0)
        self.assertEqual(schedule.placement(11).cycle, 4)
        self.assertEqual(schedule.total_cycles, 5)

    def test_form_rate_is_enforced_in_addition_to_pipe_capacity(self):
        operations = tuple(
            FloorOp(
                id=op_id,
                family="ternary",
                operands="v,v,v",
                inputs=(op_id * 3, op_id * 3 + 1, op_id * 3 + 2),
                output=18 + op_id,
                immediate=0x96,
            )
            for op_id in range(4)
        )
        schedule = plan_floor(
            FloorProgram(
                inputs=tuple(range(12)),
                outputs=tuple(range(18, 22)),
                operations=operations,
            ),
            _cpu(),
        )

        # The table says VPTERNLOG rt=1, even though its eligible port set is
        # P0123.  A port-only scheduler would incorrectly finish in one cycle.
        self.assertEqual(schedule.total_cycles, 4)
        self.assertEqual(
            [schedule.operations_at(cycle) for cycle in range(4)],
            [(0,), (1,), (2,), (3,)],
        )

    def test_unknown_timing_is_rejected_instead_of_being_guessed(self):
        program = FloorProgram(
            inputs=(0, 1),
            outputs=(2,),
            operations=(
                FloorOp(
                    id=0,
                    family="add/sub",
                    operands="v,m",
                    inputs=(0, 1),
                    output=2,
                ),
            ),
        )
        with self.assertRaisesRegex(FloorPlannerError, "unknown latency"):
            plan_floor(program, _cpu())

    def test_instruction_feature_requirements_are_checked(self):
        program = FloorProgram(
            inputs=(0, 1, 2),
            outputs=(3,),
            operations=(
                FloorOp(
                    id=0,
                    family="vnni8",
                    operands="v,v,v",
                    inputs=(0, 1, 2),
                    output=3,
                ),
            ),
        )
        with self.assertRaisesRegex(FloorPlannerError, "avx512vnni"):
            plan_floor(program, cpu_from_agner_csv(
                TABLE,
                issue_width=4,
                resources=tuple(ResourceSpec(f"P{i}") for i in range(4)),
                register_files=(RegisterFileSpec("zmm", 32, 512),),
                features=("avx512f",),
            ))

    def test_register_pressure_is_a_hard_error_in_v1(self):
        program = FloorProgram(
            inputs=(0, 1, 2, 3),
            outputs=(4, 5),
            operations=(
                FloorOp(0, "bitwise", (0, 1), 4, operands="v,v,v", asm_mnemonic="vpxorq"),
                FloorOp(1, "bitwise", (2, 3), 5, operands="v,v,v", asm_mnemonic="vpxorq"),
            ),
        )
        with self.assertRaisesRegex(FloorPlannerError, "register pressure"):
            plan_floor(program, _cpu(register_count=2))

    def test_direct_assembly_executes_without_gcc(self):
        if shutil.which("as") is None or shutil.which("ld") is None:
            self.skipTest("GNU assembler and linker are required")
        if "avx512f" not in _cpu_flags():
            self.skipTest("host does not expose AVX-512F")

        program = FloorProgram(
            inputs=(0, 1, 2, 3),
            outputs=(6,),
            operations=(
                FloorOp(0, "bitwise", (0, 1), 4, operands="v,v,v", asm_mnemonic="vpxorq"),
                FloorOp(1, "bitwise", (2, 3), 5, operands="v,v,v", asm_mnemonic="vpandq"),
                FloorOp(2, "bitwise", (4, 5), 6, operands="v,v,v", asm_mnemonic="vporq"),
            ),
        )
        schedule = plan_floor(program, _cpu())
        assembly = emit_x86_64_asm(program, schedule, function_name="floor_v1")
        self.assertIn("vpandq", assembly)
        self.assertNotIn("#include", assembly)
        self.assertNotIn("_mm512", assembly)

        with tempfile.TemporaryDirectory(prefix="stc_floor_v1_") as directory:
            root = Path(directory)
            source = root / "floor.s"
            object_file = root / "floor.o"
            shared = root / "floor.so"
            source.write_text(assembly, encoding="utf-8")
            subprocess.run(["as", "--64", "-o", str(object_file), str(source)], check=True)
            subprocess.run(["ld", "-shared", "-o", str(shared), str(object_file)], check=True)

            vector_type = ctypes.c_uint64 * 8
            input_type = ctypes.c_uint64 * (8 * 4)
            inputs = input_type(*range(1, 33))
            output = vector_type()
            expected = [
                (inputs[i] ^ inputs[8 + i]) | (inputs[16 + i] & inputs[24 + i])
                for i in range(8)
            ]
            library = ctypes.CDLL(os.fspath(shared))
            function = library.floor_v1
            function.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
            function.restype = None
            function(inputs, output)
            self.assertEqual(list(output), expected)


def _cpu_flags() -> set[str]:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return set()
    for line in text.splitlines():
        if line.startswith("flags"):
            return set(line.split(":", 1)[1].split())
    return set()
