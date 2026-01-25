from __future__ import annotations

from stc.sched.emit.base import BaseEmitter
from stc.sched.schedule import Schedule
from stc.sched.regalloc import RegAllocation


class PTXEmitter(BaseEmitter):
    @property
    def name(self) -> str:
        return "ptx"

    def emit(
        self,
        schedule: Schedule,
        allocation: RegAllocation,
        gates: list,
        input_bits: int,
        outputs: list,
        function_name: str = "circuit",
    ) -> str:
        lines: list[str] = []

        num_regs = self._count_registers(allocation, gates, input_bits)
        num_spills = len(allocation.spills)

        lines.append(f".visible .func {function_name}(")
        lines.append("    .param .u64 in_ptr,")
        lines.append("    .param .u64 out_ptr")
        lines.append(") {")

        lines.append(f"    .reg .b32 %r<{num_regs}>;")
        lines.append("    .reg .u64 %rd<4>;")

        if num_spills > 0:
            lines.append(f"    .local .b32 stack[{num_spills}];")

        lines.append("")
        lines.append("    // Load input pointer")
        lines.append("    ld.param.u64 %rd0, [in_ptr];")
        lines.append("    ld.param.u64 %rd1, [out_ptr];")
        lines.append("")

        lines.append("    // Load inputs")
        for i in range(input_bits):
            reg = allocation.reg_assignment.get(i, i)
            offset = i * 4
            lines.append(f"    ld.global.b32 %r{reg}, [%rd0+{offset}];")
        lines.append("")

        stores_by_cycle: dict[int, list[tuple[int, int, int]]] = {}
        for node, reg, cycle in allocation.stores:
            stores_by_cycle.setdefault(cycle, []).append((node, reg, cycle))

        loads_by_cycle: dict[int, list[tuple[int, int, int]]] = {}
        for node, reg, cycle in allocation.loads:
            loads_by_cycle.setdefault(cycle, []).append((node, reg, cycle))

        gates_by_cycle: dict[int, list[int]] = {}
        for g_idx, cycle in schedule.gate_cycle.items():
            gates_by_cycle.setdefault(cycle, []).append(g_idx)

        total_cycles = schedule.total_cycles
        if total_cycles > 0:
            lines.append("    // Circuit operations")

        for cycle in range(total_cycles):
            if cycle in stores_by_cycle:
                for node, reg, _ in stores_by_cycle[cycle]:
                    slot = self._get_spill_slot(node, allocation)
                    src_reg = allocation.reg_assignment.get(node, reg)
                    lines.append(f"    st.local.b32 [stack+{slot * 4}], %r{src_reg};")

            if cycle in loads_by_cycle:
                for node, reg, _ in loads_by_cycle[cycle]:
                    slot = self._get_spill_slot(node, allocation)
                    lines.append(f"    ld.local.b32 %r{reg}, [stack+{slot * 4}];")

            if cycle in gates_by_cycle:
                for g_idx in gates_by_cycle[cycle]:
                    node_idx = input_bits + g_idx
                    dst_reg = allocation.reg_assignment.get(node_idx, -1)
                    if dst_reg < 0:
                        continue

                    gate = gates[g_idx]
                    line = self._emit_gate(gate, dst_reg, allocation)
                    if line:
                        lines.append(f"    {line}")

        lines.append("")
        lines.append("    // Store outputs")
        for out_idx, output in enumerate(outputs):
            if isinstance(output, tuple):
                node_idx, inverted = output
            else:
                node_idx = output
                inverted = False

            reg = allocation.reg_assignment.get(node_idx, -1)
            if reg < 0:
                continue

            offset = out_idx * 4
            if inverted:
                temp_reg = num_regs - 1
                lines.append(f"    not.b32 %r{temp_reg}, %r{reg};")
                lines.append(f"    st.global.b32 [%rd1+{offset}], %r{temp_reg};")
            else:
                lines.append(f"    st.global.b32 [%rd1+{offset}], %r{reg};")

        lines.append("")
        lines.append("    ret;")
        lines.append("}")
        lines.append("")

        return "\n".join(lines)

    def _count_registers(
        self, allocation: RegAllocation, gates: list, input_bits: int
    ) -> int:
        max_reg = 0
        for reg in allocation.reg_assignment.values():
            max_reg = max(max_reg, reg + 1)
        max_reg = max(max_reg, input_bits)
        return max(max_reg + 1, 256)

    def _get_spill_slot(self, node: int, allocation: RegAllocation) -> int:
        try:
            return allocation.spills.index(node)
        except ValueError:
            return 0

    def _emit_gate(
        self,
        gate: tuple,
        dst_reg: int,
        allocation: RegAllocation,
    ) -> str:
        if len(gate) == 5:
            op, a, b, c, imm8 = gate
        else:
            op = gate[0]
            a = gate[1] if len(gate) > 1 else -1
            b = gate[2] if len(gate) > 2 else -1
            c = -1
            imm8 = 0

        if op == "ternary":
            a_reg = allocation.reg_assignment.get(a, -1)
            b_reg = allocation.reg_assignment.get(b, -1)
            c_reg = allocation.reg_assignment.get(c, -1)
            return f"lop3.b32 %r{dst_reg}, %r{a_reg}, %r{b_reg}, %r{c_reg}, {imm8};"

        left_reg = allocation.reg_assignment.get(a, -1)
        right_reg = allocation.reg_assignment.get(b, -1)

        if op == "xor":
            return f"xor.b32 %r{dst_reg}, %r{left_reg}, %r{right_reg};"
        elif op == "and":
            return f"and.b32 %r{dst_reg}, %r{left_reg}, %r{right_reg};"
        elif op == "or":
            return f"or.b32 %r{dst_reg}, %r{left_reg}, %r{right_reg};"
        elif op == "not":
            return f"not.b32 %r{dst_reg}, %r{left_reg};"
        elif op == "const":
            if a == 0:
                return f"mov.b32 %r{dst_reg}, 0;"
            else:
                return f"mov.b32 %r{dst_reg}, 0xFFFFFFFF;"
        else:
            return ""
