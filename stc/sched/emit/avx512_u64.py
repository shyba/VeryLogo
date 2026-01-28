from __future__ import annotations

from stc.sched.emit.base import BaseEmitter
from stc.sched.regalloc import RegAllocation
from stc.sched.schedule import Schedule


class AVX512U64Emitter(BaseEmitter):
    @property
    def name(self) -> str:
        return "avx512_u64"

    def emit(
        self,
        schedule: Schedule,
        allocation: RegAllocation,
        gates: list,
        input_bits: int,
        outputs: list,
        function_name: str = "circuit",
        io_split: tuple[int, int] | None = None,
    ) -> str:
        if allocation.num_spills:
            return self._emit_naive(gates, input_bits, outputs, function_name, io_split)

        # Convention: inputs live in fixed registers r0..r(input_bits-1).
        # Allocator-assigned registers are mapped to a disjoint namespace by
        # adding this offset to avoid clobbering live inputs.
        self._input_bits = int(input_bits)
        self._reg_offset = int(input_bits)

        input_io_words = input_bits
        output_io_words = len(outputs)
        if io_split is not None:
            input_io_words, output_io_words = io_split
        state_words = input_bits - input_io_words

        lines: list[str] = []
        lines.append("#include <immintrin.h>")
        lines.append("")
        if io_split is None:
            lines.append(f"void {function_name}(__m512i* in, __m512i* out) {{")
        else:
            lines.append(
                f"static inline void {function_name}__core(const __m512i* in_io, const __m512i* st_in, __m512i* out_io, __m512i* st_out) {{"
            )

        used_alloc_regs = set(allocation.reg_assignment.values())
        used_alloc_regs.update(reg for _, reg, _ in allocation.loads if reg >= 0)
        used_alloc_regs.update(reg for _, reg, _ in allocation.stores if reg >= 0)
        max_alloc_reg = max(used_alloc_regs) if used_alloc_regs else -1
        max_reg = (
            max(input_bits - 1, self._reg_offset + max_alloc_reg)
            if input_bits > 0
            else 0
        )
        for r in range(max_reg + 1):
            lines.append(f"    __m512i r{r};")

        lines.append("    __m512i ones = _mm512_set1_epi64(-1LL);")

        spill_slots_used: set[int] = set(range(len(allocation.spills)))
        stores_by_cycle: dict[int, list[tuple[int, int, int]]] = {}
        for node, reg, cycle in allocation.stores:
            if reg >= 0:
                reg = self._reg_offset + reg
            stores_by_cycle.setdefault(cycle, []).append((node, reg, cycle))

        loads_by_cycle: dict[int, list[tuple[int, int, int]]] = {}
        for node, reg, cycle in allocation.loads:
            if reg >= 0:
                reg = self._reg_offset + reg
            loads_by_cycle.setdefault(cycle, []).append((node, reg, cycle))
            slot = self._get_spill_slot(node, allocation)
            spill_slots_used.add(slot)

        for slot in sorted(spill_slots_used):
            lines.append(f"    __m512i stack{slot};")

        node_in_reg: dict[int, int] = {}
        reg_holds_node: dict[int, int] = {}

        def _assign_reg(node: int, reg: int) -> None:
            prev_node = reg_holds_node.get(reg)
            if prev_node is not None:
                node_in_reg.pop(prev_node, None)
            reg_holds_node[reg] = node
            node_in_reg[node] = reg

        spilled_set = set(allocation.spills)
        spill_slot: dict[int, int] = {
            node: i for i, node in enumerate(allocation.spills)
        }

        for i in range(input_bits):
            reg = i
            if io_split is None:
                lines.append(f"    r{reg} = in[{i}];")
            else:
                if i < input_io_words:
                    lines.append(f"    r{reg} = in_io[{i}];")
                else:
                    lines.append(f"    r{reg} = st_in[{i - input_io_words}];")
            _assign_reg(i, reg)
            if i in spilled_set:
                lines.append(f"    stack{spill_slot[i]} = r{reg};")

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
                    dst_reg = self._reg_offset + dst_reg
                    lines.append(
                        f"    {self._emit_gate(gates[g_idx], dst_reg, allocation, node_in_reg)}"
                    )
                    _assign_reg(node_idx, dst_reg)
                    if node_idx in spilled_set:
                        lines.append(f"    stack{spill_slot[node_idx]} = r{dst_reg};")

        for out_idx, (node_idx, inverted) in enumerate(outputs):
            expr = self._node_expr(node_idx, allocation, node_in_reg)
            if io_split is None:
                dst = f"out[{out_idx}]"
            else:
                if out_idx < output_io_words:
                    dst = f"out_io[{out_idx}]"
                else:
                    dst = f"st_out[{out_idx - output_io_words}]"
            if inverted:
                lines.append(f"    {dst} = _mm512_xor_si512({expr}, ones);")
            else:
                lines.append(f"    {dst} = {expr};")

        lines.append("}")
        lines.append("")

        if io_split is not None:
            lines.append(f"void {function_name}(__m512i* in, __m512i* out) {{")
            lines.append(
                f"    {function_name}__core(in, in + {input_io_words}, out, out + {output_io_words});"
            )
            lines.append("}")
            lines.append("")

            lines.append(
                f"void {function_name}_steps_shared(const __m512i* in_io, __m512i* out_io, const __m512i* state_in, __m512i* state_out, int steps) {{"
            )
            lines.append("    const __m512i* st_r = state_in;")
            lines.append("    __m512i* st_w = state_out;")
            lines.append("    for (int k = 0; k < steps; k++) {")
            lines.append(f"        {function_name}__core(in_io, st_r, out_io, st_w);")
            lines.append("        const __m512i* tmp = st_r;")
            lines.append("        st_r = st_w;")
            lines.append("        st_w = (__m512i*)tmp;")
            lines.append("    }")
            lines.append("    if ((steps & 1) == 0) {")
            lines.append(
                f"        for (int i = 0; i < {state_words}; i++) state_out[i] = state_in[i];"
            )
            lines.append("    }")
            lines.append("}")
            lines.append("")

        return "\n".join(lines)

    def _emit_naive(
        self,
        gates: list,
        input_bits: int,
        outputs: list,
        function_name: str,
        io_split: tuple[int, int] | None,
    ) -> str:
        input_io_words = input_bits
        output_io_words = len(outputs)
        if io_split is not None:
            input_io_words, output_io_words = io_split
        state_words = input_bits - input_io_words

        lines: list[str] = []
        lines.append("#include <immintrin.h>")
        lines.append("")
        if io_split is None:
            lines.append(f"void {function_name}(__m512i* in, __m512i* out) {{")
        else:
            lines.append(
                f"static inline void {function_name}__core(const __m512i* in_io, const __m512i* st_in, __m512i* out_io, __m512i* st_out) {{"
            )

        total_nodes = input_bits + len(gates)
        for r in range(total_nodes):
            lines.append(f"    __m512i r{r};")

        lines.append("    __m512i ones = _mm512_set1_epi64(-1LL);")

        for i in range(input_bits):
            if io_split is None:
                lines.append(f"    r{i} = in[{i}];")
            else:
                if i < input_io_words:
                    lines.append(f"    r{i} = in_io[{i}];")
                else:
                    lines.append(f"    r{i} = st_in[{i - input_io_words}];")

        for g_idx, gate in enumerate(gates):
            dst = input_bits + g_idx
            lines.append(f"    {self._emit_gate_naive(gate, dst)}")

        for out_idx, (node_idx, inverted) in enumerate(outputs):
            if io_split is None:
                dst = f"out[{out_idx}]"
            else:
                if out_idx < output_io_words:
                    dst = f"out_io[{out_idx}]"
                else:
                    dst = f"st_out[{out_idx - output_io_words}]"
            if inverted:
                lines.append(f"    {dst} = _mm512_xor_si512(r{node_idx}, ones);")
            else:
                lines.append(f"    {dst} = r{node_idx};")

        lines.append("}")
        lines.append("")

        if io_split is not None:
            lines.append(f"void {function_name}(__m512i* in, __m512i* out) {{")
            lines.append(
                f"    {function_name}__core(in, in + {input_io_words}, out, out + {output_io_words});"
            )
            lines.append("}")
            lines.append("")
            lines.append(
                f"void {function_name}_steps_shared(const __m512i* in_io, __m512i* out_io, const __m512i* state_in, __m512i* state_out, int steps) {{"
            )
            lines.append("    const __m512i* st_r = state_in;")
            lines.append("    __m512i* st_w = state_out;")
            lines.append("    for (int k = 0; k < steps; k++) {")
            lines.append(f"        {function_name}__core(in_io, st_r, out_io, st_w);")
            lines.append("        const __m512i* tmp = st_r;")
            lines.append("        st_r = st_w;")
            lines.append("        st_w = (__m512i*)tmp;")
            lines.append("    }")
            lines.append("    if ((steps & 1) == 0) {")
            lines.append(
                f"        for (int i = 0; i < {state_words}; i++) state_out[i] = state_in[i];"
            )
            lines.append("    }")
            lines.append("}")
            lines.append("")

        return "\n".join(lines)

    def _emit_gate_naive(self, gate: tuple, dst: int) -> str:
        def _src(n: int) -> str:
            return "ones" if n == -1 else f"r{n}"

        op = gate[0]
        if op == "const":
            imm = int(gate[1])
            if imm == 0:
                return f"r{dst} = _mm512_setzero_si512();"
            return f"r{dst} = _mm512_set1_epi64((long long){imm}ULL);"
        if op == "not":
            a = int(gate[1])
            return f"r{dst} = _mm512_xor_si512({_src(a)}, ones);"
        if op in ("xor", "and", "or", "add", "sub", "ult"):
            a = int(gate[1])
            b = int(gate[2])
            if op == "xor":
                return f"r{dst} = _mm512_xor_si512({_src(a)}, {_src(b)});"
            if op == "and":
                return f"r{dst} = _mm512_and_si512({_src(a)}, {_src(b)});"
            if op == "andnot":
                return f"r{dst} = _mm512_andnot_si512({_src(a)}, {_src(b)});"
            if op == "or":
                return f"r{dst} = _mm512_or_si512({_src(a)}, {_src(b)});"
            if op == "add":
                return f"r{dst} = _mm512_add_epi64({_src(a)}, {_src(b)});"
            if op == "sub":
                return f"r{dst} = _mm512_sub_epi64({_src(a)}, {_src(b)});"
            return (
                f"r{dst} = _mm512_maskz_set1_epi64("
                f"_mm512_cmp_epu64_mask({_src(a)}, {_src(b)}, _MM_CMPINT_LT), 1LL);"
            )
        if op in ("shl", "lshr"):
            a = int(gate[1])
            imm = int(gate[2])
            if op == "shl":
                return f"r{dst} = _mm512_slli_epi64({_src(a)}, {imm});"
            return f"r{dst} = _mm512_srli_epi64({_src(a)}, {imm});"
        return f"r{dst} = _mm512_setzero_si512();"

    def _emit_gate(
        self,
        gate: tuple,
        dst_reg: int,
        allocation: RegAllocation,
        node_in_reg: dict[int, int],
    ) -> str:
        op = gate[0]
        if op == "const":
            imm = int(gate[1])
            if imm == 0:
                return f"r{dst_reg} = _mm512_setzero_si512();"
            return f"r{dst_reg} = _mm512_set1_epi64((long long){imm}ULL);"
        if op == "not":
            a = int(gate[1])
            a_expr = self._node_expr(a, allocation, node_in_reg)
            return f"r{dst_reg} = _mm512_xor_si512({a_expr}, _mm512_set1_epi64(-1LL));"
        if op in ("xor", "and", "andnot", "or", "add", "sub", "ult"):
            a = int(gate[1])
            b = int(gate[2])
            a_expr = self._node_expr(a, allocation, node_in_reg)
            b_expr = self._node_expr(b, allocation, node_in_reg)
            if op == "xor":
                return f"r{dst_reg} = _mm512_xor_si512({a_expr}, {b_expr});"
            if op == "and":
                return f"r{dst_reg} = _mm512_and_si512({a_expr}, {b_expr});"
            if op == "andnot":
                return f"r{dst_reg} = _mm512_andnot_si512({a_expr}, {b_expr});"
            if op == "or":
                return f"r{dst_reg} = _mm512_or_si512({a_expr}, {b_expr});"
            if op == "add":
                return f"r{dst_reg} = _mm512_add_epi64({a_expr}, {b_expr});"
            if op == "sub":
                return f"r{dst_reg} = _mm512_sub_epi64({a_expr}, {b_expr});"
            return (
                f"r{dst_reg} = _mm512_maskz_set1_epi64("
                f"_mm512_cmp_epu64_mask({a_expr}, {b_expr}, _MM_CMPINT_LT), 1LL);"
            )
        if op in ("shl", "lshr"):
            a = int(gate[1])
            imm = int(gate[2])
            a_expr = self._node_expr(a, allocation, node_in_reg)
            if op == "shl":
                return f"r{dst_reg} = _mm512_slli_epi64({a_expr}, {imm});"
            return f"r{dst_reg} = _mm512_srli_epi64({a_expr}, {imm});"
        return f"r{dst_reg} = _mm512_setzero_si512();"

    def _node_expr(
        self, node_idx: int, allocation: RegAllocation, node_in_reg: dict[int, int]
    ) -> str:
        if node_idx == -1:
            return "ones"
        reg = node_in_reg.get(node_idx)
        if reg is None:
            if 0 <= node_idx < getattr(self, "_input_bits", 0):
                reg = node_idx
            else:
                alloc_reg = allocation.reg_assignment.get(node_idx, -1)
                reg = (
                    (getattr(self, "_reg_offset", 0) + alloc_reg)
                    if alloc_reg >= 0
                    else -1
                )
        return f"r{reg}"

    def _get_spill_slot(self, node: int, allocation: RegAllocation) -> int:
        try:
            return allocation.spills.index(node)
        except ValueError:
            return 0
