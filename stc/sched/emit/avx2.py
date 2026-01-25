from __future__ import annotations

from stc.sched.emit.base import BaseEmitter
from stc.sched.schedule import Schedule
from stc.sched.regalloc import RegAllocation


class AVX2Emitter(BaseEmitter):
    @property
    def name(self) -> str:
        return "avx2"

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
        lines.append(f"void {function_name}(__m256i* in, __m256i* out) {{")

        used_regs = set(allocation.reg_assignment.values())
        max_reg = max(used_regs) if used_regs else 0
        for r in range(max_reg + 1):
            lines.append(f"    __m256i r{r};")

        needs_ones = self._needs_ones_constant(gates, outputs)
        if needs_ones:
            lines.append("    __m256i ones = _mm256_set1_epi32(-1);")

        for i in range(input_bits):
            reg = allocation.reg_assignment.get(i, i)
            lines.append(f"    r{reg} = in[{i}];")

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
            lines.append(f"    __m256i stack{slot};")

        gates_by_cycle: dict[int, list[int]] = {}
        for g_idx, cycle in schedule.gate_cycle.items():
            gates_by_cycle.setdefault(cycle, []).append(g_idx)

        total_cycles = schedule.total_cycles
        for cycle in range(total_cycles):
            if cycle in stores_by_cycle:
                for node, reg, _ in stores_by_cycle[cycle]:
                    slot = self._get_spill_slot(node, allocation)
                    src_reg = allocation.reg_assignment.get(node, reg)
                    lines.append(f"    stack{slot} = r{src_reg};")

            if cycle in loads_by_cycle:
                for node, reg, _ in loads_by_cycle[cycle]:
                    slot = self._get_spill_slot(node, allocation)
                    lines.append(f"    r{reg} = stack{slot};")

            if cycle in gates_by_cycle:
                for g_idx in gates_by_cycle[cycle]:
                    node_idx = input_bits + g_idx
                    dst_reg = allocation.reg_assignment.get(node_idx, -1)
                    if dst_reg < 0:
                        continue

                    op, left, right = gates[g_idx]
                    line = self._emit_gate(op, left, right, dst_reg, allocation)
                    lines.append(f"    {line}")

        for out_idx, (node_idx, inverted) in enumerate(outputs):
            reg = allocation.reg_assignment.get(node_idx, -1)
            if reg < 0:
                continue
            if inverted:
                lines.append(f"    out[{out_idx}] = _mm256_xor_si256(r{reg}, ones);")
            else:
                lines.append(f"    out[{out_idx}] = r{reg};")

        lines.append("}")
        lines.append("")
        return "\n".join(lines)

    def _needs_ones_constant(self, gates: list, outputs: list) -> bool:
        for op, _, _ in gates:
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

    def _emit_gate(
        self,
        op: str,
        left: int,
        right: int,
        dst_reg: int,
        allocation: RegAllocation,
    ) -> str:
        left_reg = allocation.reg_assignment.get(left, -1)
        right_reg = allocation.reg_assignment.get(right, -1)

        if op == "xor":
            return f"r{dst_reg} = _mm256_xor_si256(r{left_reg}, r{right_reg});"
        elif op == "and":
            return f"r{dst_reg} = _mm256_and_si256(r{left_reg}, r{right_reg});"
        elif op == "or":
            return f"r{dst_reg} = _mm256_or_si256(r{left_reg}, r{right_reg});"
        elif op == "not":
            return f"r{dst_reg} = _mm256_xor_si256(r{left_reg}, ones);"
        elif op == "const":
            if left == 0:
                return f"r{dst_reg} = _mm256_setzero_si256();"
            else:
                return f"r{dst_reg} = _mm256_set1_epi32(-1);"
        else:
            return f"r{dst_reg} = _mm256_setzero_si256();"
