from __future__ import annotations

from stc.sched.emit.base import BaseEmitter
from stc.sched.regalloc import RegAllocation
from stc.sched.schedule import Schedule


class AVX512Emitter(BaseEmitter):
    @property
    def name(self) -> str:
        return "avx512"

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
        lines.append("#include <immintrin.h>")
        lines.append("")
        lines.append(f"void {function_name}(__m512i* in, __m512i* out) {{")

        used_regs = set(allocation.reg_assignment.values())
        used_regs.update(reg for _, reg, _ in allocation.loads if reg >= 0)
        used_regs.update(reg for _, reg, _ in allocation.stores if reg >= 0)
        max_reg = max([input_bits - 1, *used_regs]) if input_bits > 0 else 0
        for r in range(max_reg + 1):
            lines.append(f"    __m512i r{r};")

        needs_ones = self._needs_ones_constant(gates, outputs)
        if needs_ones:
            lines.append("    __m512i ones = _mm512_set1_epi32(-1);")

        node_in_reg: dict[int, int] = {}
        reg_holds_node: dict[int, int] = {}

        def _assign_reg(node: int, reg: int) -> None:
            prev_node = reg_holds_node.get(reg)
            if prev_node is not None:
                node_in_reg.pop(prev_node, None)
            reg_holds_node[reg] = node
            node_in_reg[node] = reg

        for i in range(input_bits):
            reg = i
            lines.append(f"    r{reg} = in[{i}];")
            _assign_reg(i, reg)

        spill_slots_used: set[int] = set()
        stores_by_cycle: dict[int, list[tuple[int, int, int]]] = {}
        for node, reg, cycle in allocation.stores:
            stores_by_cycle.setdefault(cycle, []).append((node, reg, cycle))
            slot = self._get_spill_slot(node, allocation)
            spill_slots_used.add(slot)

        loads_by_cycle: dict[int, list[tuple[int, int, int]]] = {}
        for node, reg, cycle in allocation.loads:
            loads_by_cycle.setdefault(cycle, []).append((node, reg, cycle))
            slot = self._get_spill_slot(node, allocation)
            spill_slots_used.add(slot)

        for slot in sorted(spill_slots_used):
            lines.append(f"    __m512i stack{slot};")

        gates_by_cycle: dict[int, list[int]] = {}
        for g_idx, cycle in schedule.gate_cycle.items():
            gates_by_cycle.setdefault(cycle, []).append(g_idx)

        for cycle in range(schedule.total_cycles):
            if cycle in stores_by_cycle:
                for node, reg, _ in stores_by_cycle[cycle]:
                    slot = self._get_spill_slot(node, allocation)
                    lines.append(f"    stack{slot} = r{reg};")

            if cycle in loads_by_cycle:
                for node, reg, _ in loads_by_cycle[cycle]:
                    slot = self._get_spill_slot(node, allocation)
                    lines.append(f"    r{reg} = stack{slot};")
                    _assign_reg(node, reg)

            if cycle in gates_by_cycle:
                for g_idx in gates_by_cycle[cycle]:
                    node_idx = input_bits + g_idx
                    dst_reg = allocation.reg_assignment.get(node_idx, -1)
                    if dst_reg < 0:
                        continue
                    line = self._emit_gate(gates[g_idx], dst_reg, allocation, node_in_reg)
                    lines.append(f"    {line}")
                    _assign_reg(node_idx, dst_reg)

        for out_idx, (node_idx, inverted) in enumerate(outputs):
            expr = self._node_expr(node_idx, allocation, node_in_reg)
            if inverted:
                lines.append(f"    out[{out_idx}] = _mm512_xor_si512({expr}, ones);")
            else:
                lines.append(f"    out[{out_idx}] = {expr};")

        lines.append("}")
        lines.append("")
        return "\n".join(lines)

    def _needs_ones_constant(self, gates: list, outputs: list) -> bool:
        for gate in gates:
            op = gate[0]
            if op == "not":
                return True
            if op == "const":
                return True
        for _, inverted in outputs:
            if inverted:
                return True
        return False

    def _get_spill_slot(self, node: int, allocation: RegAllocation) -> int:
        try:
            return allocation.spills.index(node)
        except ValueError:
            return 0

    def _node_expr(
        self,
        node: int,
        allocation: RegAllocation,
        node_in_reg: dict[int, int],
    ) -> str:
        reg = node_in_reg.get(node)
        if reg is not None and reg >= 0:
            return f"r{reg}"
        try:
            slot = allocation.spills.index(node)
            return f"stack{slot}"
        except ValueError:
            pass
        reg = allocation.reg_assignment.get(node, -1)
        if reg >= 0:
            return f"r{reg}"
        return "_mm512_setzero_si512()"

    def _emit_gate(
        self,
        gate: tuple,
        dst_reg: int,
        allocation: RegAllocation,
        node_in_reg: dict[int, int],
    ) -> str:
        if len(gate) == 5:
            op, a, b, c, imm8 = gate
        else:
            op = gate[0]
            a = gate[1] if len(gate) > 1 else -1
            b = gate[2] if len(gate) > 2 else -1
            c = -1
            imm8 = 0

        def _src(node: int) -> str:
            return self._node_expr(node, allocation, node_in_reg)

        if op == "ternary":
            return (
                f"r{dst_reg} = _mm512_ternarylogic_epi32({_src(a)}, {_src(b)}, {_src(c)}, {imm8});"
            )

        if op == "xor":
            return f"r{dst_reg} = _mm512_xor_si512({_src(a)}, {_src(b)});"
        if op == "and":
            return f"r{dst_reg} = _mm512_and_si512({_src(a)}, {_src(b)});"
        if op == "or":
            return f"r{dst_reg} = _mm512_or_si512({_src(a)}, {_src(b)});"
        if op == "not":
            return f"r{dst_reg} = _mm512_xor_si512({_src(a)}, ones);"
        if op == "const":
            if a == 0:
                return f"r{dst_reg} = _mm512_setzero_si512();"
            return f"r{dst_reg} = _mm512_set1_epi32(-1);"
        return f"r{dst_reg} = _mm512_setzero_si512();"

