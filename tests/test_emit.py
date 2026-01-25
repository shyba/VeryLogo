import unittest

from stc.sched import list_schedule, AVX2
from stc.sched.liveness import compute_live_ranges
from stc.sched.regalloc import allocate_registers
from stc.sched.target import PTX


def make_simple_circuit():
    """XOR of two inputs."""
    gates = [("xor", 0, 1)]
    return gates, 8, [(8, False)]


def make_chain_circuit():
    """Chain: t0 = x0^x1, t1 = t0^x2, t2 = t1^x3"""
    gates = [
        ("xor", 0, 1),
        ("xor", 8, 2),
        ("xor", 9, 3),
    ]
    return gates, 8, [(10, False)]


def make_ternary_circuit():
    """Circuit with ternary gate: lop3(a, b, c)."""
    gates = [("ternary", 0, 1, 2, 0x96)]
    return gates, 8, [(8, False)]


def schedule_and_allocate(gates, input_bits, outputs, target, num_regs=16):
    schedule = list_schedule(gates, input_bits, outputs, target)
    live_ranges = compute_live_ranges(schedule, gates, input_bits, outputs)
    allocation = allocate_registers(
        live_ranges, schedule, num_regs, gates=gates, input_bits=input_bits, outputs=outputs
    )
    return schedule, allocation


class TestAVX2Emitter(unittest.TestCase):
    def test_avx2_emitter_name(self) -> None:
        from stc.sched.emit.avx2 import AVX2Emitter

        emitter = AVX2Emitter()
        self.assertEqual(emitter.name, "avx2")

    def test_avx2_simple_circuit(self) -> None:
        from stc.sched.emit.avx2 import AVX2Emitter

        gates, input_bits, outputs = make_simple_circuit()
        schedule, allocation = schedule_and_allocate(gates, input_bits, outputs, AVX2)

        emitter = AVX2Emitter()
        code = emitter.emit(schedule, allocation, gates, input_bits, outputs)

        self.assertIn("_mm256_xor_si256", code)

    def test_avx2_chain_circuit(self) -> None:
        from stc.sched.emit.avx2 import AVX2Emitter

        gates, input_bits, outputs = make_chain_circuit()
        schedule, allocation = schedule_and_allocate(gates, input_bits, outputs, AVX2)

        emitter = AVX2Emitter()
        code = emitter.emit(schedule, allocation, gates, input_bits, outputs)

        self.assertEqual(code.count("_mm256_xor_si256"), 3)

    def test_avx2_includes_headers(self) -> None:
        from stc.sched.emit.avx2 import AVX2Emitter

        gates, input_bits, outputs = make_simple_circuit()
        schedule, allocation = schedule_and_allocate(gates, input_bits, outputs, AVX2)

        emitter = AVX2Emitter()
        code = emitter.emit(schedule, allocation, gates, input_bits, outputs)

        self.assertIn("#include <immintrin.h>", code)

    def test_avx2_function_signature(self) -> None:
        from stc.sched.emit.avx2 import AVX2Emitter

        gates, input_bits, outputs = make_simple_circuit()
        schedule, allocation = schedule_and_allocate(gates, input_bits, outputs, AVX2)

        emitter = AVX2Emitter()
        code = emitter.emit(schedule, allocation, gates, input_bits, outputs)

        self.assertIn("void", code)
        self.assertIn("__m256i", code)

    def test_avx2_loads_inputs(self) -> None:
        from stc.sched.emit.avx2 import AVX2Emitter

        gates, input_bits, outputs = make_simple_circuit()
        schedule, allocation = schedule_and_allocate(gates, input_bits, outputs, AVX2)

        emitter = AVX2Emitter()
        code = emitter.emit(schedule, allocation, gates, input_bits, outputs)

        self.assertIn("in[", code)

    def test_avx2_stores_outputs(self) -> None:
        from stc.sched.emit.avx2 import AVX2Emitter

        gates, input_bits, outputs = make_simple_circuit()
        schedule, allocation = schedule_and_allocate(gates, input_bits, outputs, AVX2)

        emitter = AVX2Emitter()
        code = emitter.emit(schedule, allocation, gates, input_bits, outputs)

        self.assertIn("out[", code)

    def test_avx2_handles_spills(self) -> None:
        from stc.sched.emit.avx2 import AVX2Emitter

        gates = [("xor", i, i + 1) for i in range(15)]
        input_bits = 16
        outputs = [(input_bits + 14, False)]

        schedule, allocation = schedule_and_allocate(
            gates, input_bits, outputs, AVX2, num_regs=4
        )

        emitter = AVX2Emitter()
        code = emitter.emit(schedule, allocation, gates, input_bits, outputs)

        self.assertTrue(
            "out[" in code or "stack" in code,
            "Expected output or spill code in output",
        )


class TestPTXEmitter(unittest.TestCase):
    def test_ptx_emitter_name(self) -> None:
        from stc.sched.emit.ptx import PTXEmitter

        emitter = PTXEmitter()
        self.assertEqual(emitter.name, "ptx")

    def test_ptx_simple_circuit(self) -> None:
        from stc.sched.emit.ptx import PTXEmitter

        gates, input_bits, outputs = make_simple_circuit()
        schedule, allocation = schedule_and_allocate(gates, input_bits, outputs, PTX)

        emitter = PTXEmitter()
        code = emitter.emit(schedule, allocation, gates, input_bits, outputs)

        self.assertIn("xor.b32", code)

    def test_ptx_chain_circuit(self) -> None:
        from stc.sched.emit.ptx import PTXEmitter

        gates, input_bits, outputs = make_chain_circuit()
        schedule, allocation = schedule_and_allocate(gates, input_bits, outputs, PTX)

        emitter = PTXEmitter()
        code = emitter.emit(schedule, allocation, gates, input_bits, outputs)

        self.assertEqual(code.count("xor.b32"), 3)

    def test_ptx_function_signature(self) -> None:
        from stc.sched.emit.ptx import PTXEmitter

        gates, input_bits, outputs = make_simple_circuit()
        schedule, allocation = schedule_and_allocate(gates, input_bits, outputs, PTX)

        emitter = PTXEmitter()
        code = emitter.emit(schedule, allocation, gates, input_bits, outputs)

        self.assertIn(".visible .func", code)

    def test_ptx_register_declaration(self) -> None:
        from stc.sched.emit.ptx import PTXEmitter

        gates, input_bits, outputs = make_simple_circuit()
        schedule, allocation = schedule_and_allocate(gates, input_bits, outputs, PTX)

        emitter = PTXEmitter()
        code = emitter.emit(schedule, allocation, gates, input_bits, outputs)

        self.assertIn(".reg .b32", code)

    def test_ptx_loads_inputs(self) -> None:
        from stc.sched.emit.ptx import PTXEmitter

        gates, input_bits, outputs = make_simple_circuit()
        schedule, allocation = schedule_and_allocate(gates, input_bits, outputs, PTX)

        emitter = PTXEmitter()
        code = emitter.emit(schedule, allocation, gates, input_bits, outputs)

        self.assertIn("ld.global.b32", code)

    def test_ptx_stores_outputs(self) -> None:
        from stc.sched.emit.ptx import PTXEmitter

        gates, input_bits, outputs = make_simple_circuit()
        schedule, allocation = schedule_and_allocate(gates, input_bits, outputs, PTX)

        emitter = PTXEmitter()
        code = emitter.emit(schedule, allocation, gates, input_bits, outputs)

        self.assertIn("st.global.b32", code)

    def test_ptx_handles_ternary(self) -> None:
        from stc.sched.emit.ptx import PTXEmitter

        gates, input_bits, outputs = make_ternary_circuit()
        schedule, allocation = schedule_and_allocate(gates, input_bits, outputs, PTX)

        emitter = PTXEmitter()
        code = emitter.emit(schedule, allocation, gates, input_bits, outputs)

        self.assertIn("lop3.b32", code)
