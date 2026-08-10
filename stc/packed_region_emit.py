from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stc.packed_circuit import PackedCircuitState, PackedGate
from stc.packed_regions import RegionCaps, build_dep_graph, partition_into_regions
from stc.tick_ir_to_packed_circuit_state import PackedWordLayout


@dataclass(frozen=True)
class EmitRegionsConfig:
    caps: RegionCaps
    diagnostics_dir: Path | None = None


def emit_avx512_u64_regions(
    circuit: PackedCircuitState,
    layout: PackedWordLayout,
    *,
    function_name: str = "circuit",
    config: EmitRegionsConfig | None = None,
) -> str:
    """
    Emit AVX-512 (epi64) code using region decomposition.

    This is a “compiler-level” approach: it breaks the tick into regions and
    materializes only region interfaces (packed words), enabling the C compiler
    to keep many values register-resident within each region and limiting peak
    liveness across regions.

    The ABI mirrors the split-IO convention used by the bit-level AVX emitters:
      - core(in_io, st_in, out_io, st_out)
      - steps_shared(in_io, out_io, state_in, state_out, steps)
    """
    if circuit.word_bits != 64:
        raise ValueError("emit_avx512_u64_regions requires 64-bit packed circuit")

    if config is None:
        config = EmitRegionsConfig(
            caps=RegionCaps(max_gates=50_000, max_boundary=8_192)
        )

    graph = build_dep_graph(circuit)
    regions = partition_into_regions(graph, config.caps)

    if config.diagnostics_dir is not None:
        from stc.region_diagnostics import (
            generate_region_diagnostics,
            write_region_diagnostics,
            write_region_dot,
        )

        diagnostics = generate_region_diagnostics(regions, graph)
        write_region_diagnostics(diagnostics, config.diagnostics_dir)
        write_region_dot(regions, graph, config.diagnostics_dir)

    input_io_words, output_io_words = layout.io_words()
    state_words = layout.input_words - input_io_words

    # Boundary nodes: region outputs that are not primary outputs / next_state.
    primary_out_nodes = {n for (n, _inv) in circuit.outputs}
    boundary_nodes: list[int] = []
    boundary_set: set[int] = set()
    for r in regions:
        for node in r.outputs:
            if node in primary_out_nodes:
                continue
            if node not in boundary_set:
                boundary_set.add(node)
                boundary_nodes.append(node)
    boundary_nodes.sort()
    boundary_slot = {n: i for i, n in enumerate(boundary_nodes)}

    # Map node index -> producing gate index (for gate outputs only).
    prod_gate = {graph.gate_output_node(gi): gi for gi in range(len(circuit.gates))}

    lines: list[str] = []
    lines.append("#include <immintrin.h>")
    lines.append("")
    lines.append(
        f"static inline void {function_name}__core(const __m512i* in_io, const __m512i* st_in, __m512i* out_io, __m512i* st_out) {{"
    )
    if boundary_nodes:
        lines.append(f"    __m512i b[{len(boundary_nodes)}];")

    lines.append("    const __m512i ones = _mm512_set1_epi64(-1LL);")

    def src_expr(node: int, local_map: dict[int, str]) -> str:
        if node in local_map:
            return local_map[node]
        if node < layout.input_words:
            if node < input_io_words:
                return f"in_io[{node}]"
            return f"st_in[{node - input_io_words}]"
        # Produced by an earlier region => boundary
        slot = boundary_slot.get(node)
        if slot is None:
            raise ValueError(f"missing boundary slot for node {node}")
        return f"b[{slot}]"

    def emit_gate(g: PackedGate, dst: str, get: callable) -> str:
        op = g[0]
        if op == "const":
            imm = int(g[1]) & ((1 << 64) - 1)
            if imm == 0:
                return f"{dst} = _mm512_setzero_si512();"
            return f"{dst} = _mm512_set1_epi64((long long){imm}ULL);"
        if op == "not":
            a = get(int(g[1]))
            return f"{dst} = _mm512_xor_si512({a}, ones);"
        if op in ("xor", "and", "andnot", "or", "add", "sub"):
            a = get(int(g[1]))
            b = get(int(g[2]))
            if op == "xor":
                return f"{dst} = _mm512_xor_si512({a}, {b});"
            if op == "and":
                return f"{dst} = _mm512_and_si512({a}, {b});"
            if op == "andnot":
                return f"{dst} = _mm512_andnot_si512({a}, {b});"
            if op == "or":
                return f"{dst} = _mm512_or_si512({a}, {b});"
            if op == "add":
                return f"{dst} = _mm512_add_epi64({a}, {b});"
            return f"{dst} = _mm512_sub_epi64({a}, {b});"
        if op == "ternary":
            a = get(int(g[1]))
            b = get(int(g[2]))
            c = get(int(g[3]))
            imm8 = int(g[4]) & 0xFF
            return f"{dst} = _mm512_ternarylogic_epi64({a}, {b}, {c}, {imm8});"
        if op in ("shl", "lshr"):
            a = get(int(g[1]))
            imm = int(g[2])
            if op == "shl":
                return f"{dst} = _mm512_slli_epi64({a}, {imm});"
            return f"{dst} = _mm512_srli_epi64({a}, {imm});"
        raise ValueError(f"unsupported packed op in emitter: {op}")

    # Emit each region as a block with a compact local value array.
    for r in regions:
        lines.append(
            f"    // region {r.id}: {len(r.gate_indices)} gates, {len(r.inputs)} in, {len(r.outputs)} out"
        )

        # Local value slots:
        # - first for each region input
        # - then for each gate output in this region
        local_map: dict[int, str] = {}

        if r.inputs:
            lines.append(f"    __m512i rin_{r.id}[{len(r.inputs)}];")
            for i, node in enumerate(r.inputs):
                local_map[node] = f"rin_{r.id}[{i}]"
                lines.append(f"    rin_{r.id}[{i}] = {src_expr(node, {})};")

        lines.append(f"    __m512i rtmp_{r.id}[{len(r.gate_indices)}];")
        gate_index_to_local: dict[int, int] = {}
        for li, gi in enumerate(r.gate_indices):
            node = graph.gate_output_node(gi)
            gate_index_to_local[gi] = li
            local_map[node] = f"rtmp_{r.id}[{li}]"

        def get_local(node: int) -> str:
            return src_expr(node, local_map)

        for gi in r.gate_indices:
            li = gate_index_to_local[gi]
            dst = f"rtmp_{r.id}[{li}]"
            g = circuit.gates[gi]
            lines.append(f"    {emit_gate(g, dst, get_local)}")

        # Store region outputs to either final outputs/next_state or boundary array.
        out_set = set(r.outputs)
        for node in r.outputs:
            val = src_expr(node, local_map)
            # Is it a primary output? Determine index in circuit.outputs.
            if node in primary_out_nodes:
                # Find all output indices for this node (it can appear multiple times).
                for out_idx, (onode, inv) in enumerate(circuit.outputs):
                    if onode != node:
                        continue
                    if out_idx < output_io_words:
                        dst = f"out_io[{out_idx}]"
                    else:
                        dst = f"st_out[{out_idx - output_io_words}]"
                    if inv:
                        lines.append(f"    {dst} = _mm512_xor_si512({val}, ones);")
                    else:
                        lines.append(f"    {dst} = {val};")
            else:
                slot = boundary_slot[node]
                lines.append(f"    b[{slot}] = {val};")

    lines.append("}")
    lines.append("")

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
