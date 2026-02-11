from __future__ import annotations

from dataclasses import dataclass

from stc.control_specialize import (
    compute_const_state_schedule,
    const_state_repl_for_step,
    specialize_expr,
)
from stc.interp import infer_type
from stc.tick_ir import (
    AShr,
    Add,
    And,
    Bitcast,
    BitVecConst,
    BitVecType,
    BoolConst,
    BoolType,
    Concat,
    Eq,
    Expr,
    LShr,
    Lut8,
    Mux,
    Not,
    Or,
    Shl,
    Slice,
    Sub,
    TickIR,
    Type,
    Uge,
    Ugt,
    Ule,
    Ult,
    Var,
    Xor,
)
from stc.tick_ir_validate import validate_tick_ir


@dataclass(frozen=True)
class CodegenError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class _V:
    kind: str
    width: int | None = None
    words: list[str] | None = None
    pred: str | None = None


def _words_for_width(width: int) -> int:
    if width < 1:
        raise CodegenError("bitvec width must be >= 1")
    return (width + 31) // 32


def _word_mask(width: int, word_index: int) -> int:
    if width < 1:
        raise CodegenError("bitvec width must be >= 1")
    n = _words_for_width(width)
    if word_index < 0 or word_index >= n:
        raise CodegenError("word index out of range")
    if word_index != n - 1:
        return 0xFFFFFFFF
    rem = width % 32
    if rem == 0:
        return 0xFFFFFFFF
    return (1 << rem) - 1


def _port_words(t: Type) -> int:
    if isinstance(t, BoolType):
        return 1
    if isinstance(t, BitVecType):
        return _words_for_width(t.width)
    raise CodegenError("ptx backend supports only bool and bitvec types")


def _port_offsets(ts: dict[str, Type], order: list[str]) -> tuple[dict[str, int], int]:
    off = 0
    offsets: dict[str, int] = {}
    for name in order:
        offsets[name] = off
        off += _port_words(ts[name])
    return offsets, off


def emit_ptx(ir: TickIR, *, sm: str = "sm_61") -> str:
    return emit_ptx_steps(ir, sm=sm, steps=1)


def emit_ptx_steps(
    ir: TickIR,
    *,
    sm: str = "sm_61",
    steps: int = 1,
    assume_reset_state: bool = False,
    runtime_step_loop: bool = False,
) -> str:
    validate_tick_ir(ir)
    if int(steps) < 1:
        raise CodegenError("steps must be >= 1")

    for t in (
        list(ir.inputs.values()) + list(ir.outputs.values()) + list(ir.state.values())
    ):
        _port_words(t)

    ctx_types: dict[str, Type] = {**ir.inputs, **ir.state}

    input_order = sorted(ir.inputs.keys())
    output_order = sorted(ir.outputs.keys())
    state_order = sorted(ir.state.keys())

    schedule = (
        compute_const_state_schedule(ir, steps=int(steps))
        if assume_reset_state and ir.state
        else None
    )
    const_state_steps = schedule.const_state_steps if schedule is not None else []
    const_state_final = schedule.const_state_final if schedule is not None else {}
    const_state_order = schedule.const_state_order if schedule is not None else []
    runtime_state_order = (
        schedule.runtime_state_order if schedule is not None else list(state_order)
    )
    reset_vals_all = schedule.reset_vals if schedule is not None else {}

    in_offs, in_stride = _port_offsets(ir.inputs, input_order)
    out_offs, out_stride = _port_offsets(ir.outputs, output_order)
    st_offs, st_stride = _port_offsets(ir.state, state_order) if ir.state else ({}, 0)

    r_cnt = 0
    rd_cnt = 0
    p_cnt = 0
    lbl_cnt = 0

    lines: list[str] = []
    body: list[str] = []
    memo: dict[int, _V] = {}

    lut_decls: list[str] = []
    lut_syms: dict[bytes, str] = {}

    def new_r() -> str:
        nonlocal r_cnt
        name = f"%r{r_cnt}"
        r_cnt += 1
        return name

    def new_rd() -> str:
        nonlocal rd_cnt
        name = f"%rd{rd_cnt}"
        rd_cnt += 1
        return name

    def new_p() -> str:
        nonlocal p_cnt
        name = f"%p{p_cnt}"
        p_cnt += 1
        return name

    def new_lbl(prefix: str) -> str:
        nonlocal lbl_cnt
        name = f"{prefix}_{lbl_cnt}"
        lbl_cnt += 1
        return name

    def emit_mask_word(word: str, mask: int) -> str:
        if mask == 0xFFFFFFFF:
            return word
        out = new_r()
        body.append(f"  and.b32 {out}, {word}, 0x{mask:x};")
        return out

    def emit_mask_value(v: _V) -> _V:
        if v.kind != "bits" or v.words is None or v.width is None:
            raise CodegenError("expected bits value")
        n = _words_for_width(v.width)
        out_words: list[str] = []
        for i in range(n):
            mask = _word_mask(v.width, i)
            out_words.append(emit_mask_word(v.words[i], mask))
        return _V(kind="bits", width=v.width, words=out_words)

    def emit_bool_to_bits(p: str, width: int) -> _V:
        n = _words_for_width(width)
        w0 = new_r()
        body.append(f"  selp.b32 {w0}, 1, 0, {p};")
        out = [w0]
        for _ in range(1, n):
            z = new_r()
            body.append(f"  mov.u32 {z}, 0;")
            out.append(z)
        v = _V(kind="bits", width=width, words=out)
        return emit_mask_value(v)

    def emit_bits_to_bool(v: _V) -> str:
        if v.kind != "bits" or v.words is None or v.width is None:
            raise CodegenError("expected bits value")
        w0 = v.words[0]
        p = new_p()
        body.append(f"  setp.ne.u32 {p}, {w0}, 0;")
        return p

    def emit_addr(base_rd: str, idx_r: str, stride_words: int, off_words: int) -> str:
        elem = new_r()
        body.append(f"  mad.lo.u32 {elem}, {idx_r}, {stride_words}, {off_words};")
        off_rd = new_rd()
        body.append(f"  mul.wide.u32 {off_rd}, {elem}, 4;")
        addr = new_rd()
        body.append(f"  add.u64 {addr}, {base_rd}, {off_rd};")
        return addr

    def emit_load_u32(addr: str) -> str:
        r = new_r()
        body.append(f"  ld.global.u32 {r}, [{addr}];")
        return r

    def emit_store_u32(addr: str, r: str) -> None:
        body.append(f"  st.global.u32 [{addr}], {r};")

    def emit_load_port(
        t: Type,
        idx_r: str,
        base_rd: str,
        stride_words: int,
        off_words: int,
    ) -> _V:
        if isinstance(t, BoolType):
            addr = emit_addr(base_rd, idx_r, stride_words, off_words)
            r = emit_load_u32(addr)
            p = new_p()
            body.append(f"  setp.ne.u32 {p}, {r}, 0;")
            return _V(kind="pred", pred=p)

        assert isinstance(t, BitVecType)
        n = _words_for_width(t.width)
        words: list[str] = []
        for i in range(n):
            addr = emit_addr(base_rd, idx_r, stride_words, off_words + i)
            words.append(emit_load_u32(addr))
        v = _V(kind="bits", width=t.width, words=words)
        return emit_mask_value(v)

    def emit_store_port(
        v: _V,
        t: Type,
        idx_r: str,
        base_rd: str,
        stride_words: int,
        off_words: int,
    ) -> None:
        if isinstance(t, BoolType):
            if v.kind != "pred" or v.pred is None:
                raise CodegenError("bool store requires pred value")
            addr = emit_addr(base_rd, idx_r, stride_words, off_words)
            r = new_r()
            body.append(f"  selp.b32 {r}, 1, 0, {v.pred};")
            emit_store_u32(addr, r)
            return

        assert isinstance(t, BitVecType)
        if v.kind == "pred" and v.pred is not None:
            v = emit_bool_to_bits(v.pred, t.width)
        if v.kind != "bits" or v.words is None or v.width is None:
            raise CodegenError("bitvec store requires bits value")
        if v.width != t.width:
            raise CodegenError("store width mismatch")
        n = _words_for_width(t.width)
        v = emit_mask_value(v)
        for i in range(n):
            addr = emit_addr(base_rd, idx_r, stride_words, off_words + i)
            emit_store_u32(addr, v.words[i])

    def emit_copy_value(dst: _V, src: _V, t: Type) -> None:
        """Copy src into dst registers in-place (for runtime state loops)."""
        if isinstance(t, BoolType):
            if dst.kind != "pred" or dst.pred is None:
                raise CodegenError("bool state destination must be pred")
            if src.kind != "pred" or src.pred is None:
                raise CodegenError("bool state source must be pred")
            body.append(f"  mov.pred {dst.pred}, {src.pred};")
            return

        assert isinstance(t, BitVecType)
        if dst.kind != "bits" or dst.words is None or dst.width != t.width:
            raise CodegenError("bitvec state destination mismatch")
        if src.kind == "pred" and src.pred is not None:
            src = emit_bool_to_bits(src.pred, t.width)
        if src.kind != "bits" or src.words is None or src.width != t.width:
            raise CodegenError("bitvec state source mismatch")
        src = emit_mask_value(src)
        for dw, sw in zip(dst.words, src.words):
            body.append(f"  mov.b32 {dw}, {sw};")

    env: dict[str, _V] = {}

    def emit_const_u32(x: int) -> str:
        r = new_r()
        body.append(f"  mov.u32 {r}, {x};")
        return r

    def lut8_symbol(table: list[int]) -> str:
        key = bytes(int(v) & 0xFF for v in table)
        sym = lut_syms.get(key)
        if sym is not None:
            return sym
        sym = f"__stc_lut8_{len(lut_syms)}"
        lut_syms[key] = sym
        lut_decls.append(f".visible .const .align 1 .b8 {sym}[256] = {{")
        for off in range(0, 256, 16):
            chunk = ", ".join(f"0x{key[i]:02x}" for i in range(off, off + 16))
            comma = "," if off + 16 < 256 else ""
            lut_decls.append(f"  {chunk}{comma}")
        lut_decls.append("};")
        lut_decls.append("")
        return sym

    def emit_expr(expr: Expr) -> _V:
        key = id(expr)
        if key in memo:
            return memo[key]

        if isinstance(expr, Var):
            v = env.get(expr.name)
            if v is None:
                raise CodegenError(f"missing var {expr.name}")
            memo[key] = v
            return v

        if isinstance(expr, BoolConst):
            p = new_p()
            body.append(f"  mov.pred {p}, {1 if expr.value else 0};")
            v = _V(kind="pred", pred=p)
            memo[key] = v
            return v

        if isinstance(expr, BitVecConst):
            n = _words_for_width(expr.width)
            words: list[str] = []
            for i in range(n):
                w = (expr.value >> (32 * i)) & 0xFFFFFFFF
                r = new_r()
                body.append(f"  mov.u32 {r}, 0x{w:x};")
                words.append(r)
            v = emit_mask_value(_V(kind="bits", width=expr.width, words=words))
            memo[key] = v
            return v

        if isinstance(expr, Bitcast):
            dst = expr.to
            if isinstance(dst, BoolType):
                raise CodegenError("bitcast to bool is not supported in PTX backend")
            if not isinstance(dst, BitVecType):
                raise CodegenError("ptx backend supports only bitcast to bitvec")
            x = emit_expr(expr.x)
            if x.kind == "pred" and x.pred is not None:
                x = emit_bool_to_bits(x.pred, dst.width)
            if x.kind != "bits" or x.words is None or x.width is None:
                raise CodegenError("bitcast requires bits operand")
            src_words = _words_for_width(x.width)
            dst_words = _words_for_width(dst.width)
            out: list[str] = []
            for i in range(dst_words):
                if i < src_words:
                    out.append(x.words[i])
                else:
                    z = new_r()
                    body.append(f"  mov.u32 {z}, 0;")
                    out.append(z)
            v = emit_mask_value(_V(kind="bits", width=dst.width, words=out))
            memo[key] = v
            return v

        if isinstance(expr, Lut8):
            x_t = infer_type(expr.x, ctx_types)
            if not isinstance(x_t, BitVecType) or x_t.width != 8:
                raise CodegenError("lut8 input must be bitvec8")
            x = emit_expr(expr.x)
            if x.kind == "pred" and x.pred is not None:
                x = emit_bool_to_bits(x.pred, 8)
            if x.kind != "bits" or x.words is None or x.width is None:
                raise CodegenError("lut8 input must be bits")
            if x.width != 8:
                raise CodegenError("lut8 input width mismatch")

            sym = lut8_symbol(expr.table)
            idx = new_r()
            body.append(f"  and.b32 {idx}, {x.words[0]}, 0xff;")
            rd_base = new_rd()
            body.append(f"  mov.u64 {rd_base}, {sym};")
            rd_off = new_rd()
            body.append(f"  cvt.u64.u32 {rd_off}, {idx};")
            rd_addr = new_rd()
            body.append(f"  add.u64 {rd_addr}, {rd_base}, {rd_off};")
            r = new_r()
            body.append(f"  ld.const.u8 {r}, [{rd_addr}];")
            v = emit_mask_value(_V(kind="bits", width=8, words=[r]))
            memo[key] = v
            return v

        from stc.tick_ir import TernaryLut

        if isinstance(expr, TernaryLut):
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            c = emit_expr(expr.c)

            if a.kind == "pred" and a.pred is not None:
                a_t = infer_type(expr.a, ctx_types)
                if isinstance(a_t, BoolType):
                    a = emit_bool_to_bits(a.pred, 1)

            if not (a.kind == "bits" and a.words):
                raise CodegenError("TernaryLut input must be bits")

            width = a.width or 1
            result_words = []

            for word_idx in range(len(a.words)):
                a_word = a.words[word_idx] if word_idx < len(a.words) else "0"
                b_word = (
                    b.words[word_idx] if b.words and word_idx < len(b.words) else "0"
                )
                c_word = (
                    c.words[word_idx] if c.words and word_idx < len(c.words) else "0"
                )

                r = new_r()
                body.append(
                    f"  lop3.b32 {r}, {a_word}, {b_word}, {c_word}, {expr.imm8};"
                )
                result_words.append(r)

            v = _V(kind="bits", width=width, words=result_words)
            memo[key] = v
            return v

        if isinstance(expr, Not):
            t = infer_type(expr.x, ctx_types)
            x = emit_expr(expr.x)
            if isinstance(t, BoolType):
                if x.kind != "pred" or x.pred is None:
                    raise CodegenError("bool not requires pred operand")
                p = new_p()
                body.append(f"  not.pred {p}, {x.pred};")
                v = _V(kind="pred", pred=p)
                memo[key] = v
                return v
            assert isinstance(t, BitVecType)
            if x.kind == "pred" and x.pred is not None:
                x = emit_bool_to_bits(x.pred, t.width)
            if x.kind != "bits" or x.words is None or x.width is None:
                raise CodegenError("bitvec not requires bits operand")
            if x.width != t.width:
                raise CodegenError("not width mismatch")
            n = _words_for_width(t.width)
            out: list[str] = []
            for i in range(n):
                r = new_r()
                body.append(f"  not.b32 {r}, {x.words[i]};")
                out.append(r)
            v = emit_mask_value(_V(kind="bits", width=t.width, words=out))
            memo[key] = v
            return v

        if isinstance(expr, (And, Or, Xor)):
            t = infer_type(expr.a, ctx_types)
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            if isinstance(t, BoolType):
                if (
                    a.kind != "pred"
                    or a.pred is None
                    or b.kind != "pred"
                    or b.pred is None
                ):
                    raise CodegenError("bool op requires pred operands")
                p = new_p()
                op = (
                    "and"
                    if isinstance(expr, And)
                    else "or" if isinstance(expr, Or) else "xor"
                )
                body.append(f"  {op}.pred {p}, {a.pred}, {b.pred};")
                v = _V(kind="pred", pred=p)
                memo[key] = v
                return v

            assert isinstance(t, BitVecType)
            if a.kind == "pred" and a.pred is not None:
                a = emit_bool_to_bits(a.pred, t.width)
            if b.kind == "pred" and b.pred is not None:
                b = emit_bool_to_bits(b.pred, t.width)
            if (
                a.kind != "bits"
                or b.kind != "bits"
                or a.words is None
                or b.words is None
            ):
                raise CodegenError("bitvec op requires bits operands")
            if a.width != t.width or b.width != t.width:
                raise CodegenError("bitvec op width mismatch")

            n = _words_for_width(t.width)
            out: list[str] = []
            op = (
                "and"
                if isinstance(expr, And)
                else "or" if isinstance(expr, Or) else "xor"
            )
            for i in range(n):
                r = new_r()
                body.append(f"  {op}.b32 {r}, {a.words[i]}, {b.words[i]};")
                out.append(r)
            v = emit_mask_value(_V(kind="bits", width=t.width, words=out))
            memo[key] = v
            return v

        if isinstance(expr, (Add, Sub)):
            t = infer_type(expr.a, ctx_types)
            assert isinstance(t, BitVecType)
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            if a.kind == "pred" and a.pred is not None:
                a = emit_bool_to_bits(a.pred, t.width)
            if b.kind == "pred" and b.pred is not None:
                b = emit_bool_to_bits(b.pred, t.width)
            if (
                a.kind != "bits"
                or b.kind != "bits"
                or a.words is None
                or b.words is None
            ):
                raise CodegenError("add/sub requires bits operands")
            if a.width != t.width or b.width != t.width:
                raise CodegenError("add/sub width mismatch")

            n = _words_for_width(t.width)
            out: list[str] = []
            if n == 1:
                r = new_r()
                op = "add" if isinstance(expr, Add) else "sub"
                body.append(f"  {op}.u32 {r}, {a.words[0]}, {b.words[0]};")
                out.append(r)
            else:
                r0 = new_r()
                op = "add" if isinstance(expr, Add) else "sub"
                body.append(f"  {op}.cc.u32 {r0}, {a.words[0]}, {b.words[0]};")
                out.append(r0)
                for i in range(1, n):
                    ri = new_r()
                    opc = "addc" if isinstance(expr, Add) else "subc"
                    body.append(f"  {opc}.u32 {ri}, {a.words[i]}, {b.words[i]};")
                    out.append(ri)
            v = emit_mask_value(_V(kind="bits", width=t.width, words=out))
            memo[key] = v
            return v

        if isinstance(expr, (Shl, LShr, AShr)):
            t = infer_type(expr.a, ctx_types)
            assert isinstance(t, BitVecType)
            x = emit_expr(expr.a)
            sh = emit_expr(expr.b)
            if x.kind == "pred" and x.pred is not None:
                x = emit_bool_to_bits(x.pred, t.width)
            if x.kind != "bits" or x.words is None or x.width is None:
                raise CodegenError("shift requires bits lhs")
            if x.width != t.width:
                raise CodegenError("shift width mismatch")
            if sh.kind == "pred" and sh.pred is not None:
                sh = emit_bool_to_bits(sh.pred, 32)
            if sh.kind != "bits" or sh.words is None:
                raise CodegenError("shift amount must be bitvec")

            sh0 = sh.words[0]
            word_shift = new_r()
            bit_shift = new_r()
            body.append(f"  shr.u32 {word_shift}, {sh0}, 5;")
            body.append(f"  and.b32 {bit_shift}, {sh0}, 31;")
            p_nz = new_p()
            body.append(f"  setp.ne.u32 {p_nz}, {bit_shift}, 0;")
            inv_bit = new_r()
            body.append(f"  sub.u32 {inv_bit}, 32, {bit_shift};")

            n = _words_for_width(t.width)
            words: list[str] = []

            if isinstance(expr, AShr):
                top = x.words[n - 1]
                sign_pos = (t.width - 1) % 32
                tmp = new_r()
                body.append(f"  shr.u32 {tmp}, {top}, {sign_pos};")
                tmp2 = new_r()
                body.append(f"  and.b32 {tmp2}, {tmp}, 1;")
                p_sign = new_p()
                body.append(f"  setp.ne.u32 {p_sign}, {tmp2}, 0;")
                mask_unused = (~_word_mask(t.width, n - 1)) & 0xFFFFFFFF
                top_or = new_r()
                body.append(f"  or.b32 {top_or}, {top}, 0x{mask_unused:x};")
                ext_top = new_r()
                body.append(f"  selp.b32 {ext_top}, {top_or}, {top}, {p_sign};")
                fill = new_r()
                body.append(f"  selp.b32 {fill}, 0xffffffff, 0, {p_sign};")
                src_words = list(x.words[:-1]) + [ext_top]
            else:
                fill = None
                src_words = x.words

            def select_idx(idx_reg: str, default_word: str) -> str:
                cur = default_word
                for j, wj in enumerate(src_words):
                    p = new_p()
                    body.append(f"  setp.eq.u32 {p}, {idx_reg}, {j};")
                    nxt = new_r()
                    body.append(f"  selp.b32 {nxt}, {wj}, {cur}, {p};")
                    cur = nxt
                return cur

            for i in range(n):
                if isinstance(expr, Shl):
                    if i == 0:
                        p_ok = new_p()
                        body.append(f"  setp.eq.u32 {p_ok}, {word_shift}, 0;")
                        src_idx = emit_const_u32(0)
                    else:
                        p_ok = new_p()
                        body.append(f"  setp.le.u32 {p_ok}, {word_shift}, {i};")
                        src_idx = new_r()
                        body.append(f"  sub.u32 {src_idx}, {i}, {word_shift};")
                    cur = select_idx(src_idx, emit_const_u32(0))
                    cur_sel = new_r()
                    body.append(f"  selp.b32 {cur_sel}, {cur}, 0, {p_ok};")
                    res = new_r()
                    body.append(f"  mov.u32 {res}, {cur_sel};")

                    if i == 0:
                        prev_sel = emit_const_u32(0)
                    else:
                        p_prev_ok = new_p()
                        body.append(
                            f"  setp.le.u32 {p_prev_ok}, {word_shift}, {i - 1};"
                        )
                        prev_idx = new_r()
                        body.append(f"  sub.u32 {prev_idx}, {i - 1}, {word_shift};")
                        prev = select_idx(prev_idx, emit_const_u32(0))
                        prev_sel = new_r()
                        body.append(f"  selp.b32 {prev_sel}, {prev}, 0, {p_prev_ok};")

                    lo = new_r()
                    body.append(f"  mov.u32 {lo}, 0;")
                    hi = new_r()
                    body.append(f"  mov.u32 {hi}, 0;")
                    body.append(f"  @{p_nz} shl.b32 {lo}, {cur_sel}, {bit_shift};")
                    body.append(f"  @{p_nz} shr.u32 {hi}, {prev_sel}, {inv_bit};")
                    comb = new_r()
                    body.append(f"  or.b32 {comb}, {lo}, {hi};")
                    body.append(f"  @{p_nz} mov.u32 {res}, {comb};")

                    words.append(res)

                else:
                    base = new_r()
                    body.append(f"  add.u32 {base}, {word_shift}, {i};")
                    default_word = fill if fill is not None else emit_const_u32(0)
                    cur = select_idx(base, default_word)

                    cur_sel = new_r()
                    body.append(f"  mov.u32 {cur_sel}, {cur};")

                    nxt_idx = new_r()
                    body.append(f"  add.u32 {nxt_idx}, {base}, 1;")
                    nxt = select_idx(nxt_idx, default_word)

                    res = new_r()
                    body.append(f"  mov.u32 {res}, {cur_sel};")

                    lo = new_r()
                    body.append(f"  mov.u32 {lo}, 0;")
                    hi = new_r()
                    body.append(f"  mov.u32 {hi}, 0;")
                    if isinstance(expr, LShr):
                        body.append(f"  @{p_nz} shr.u32 {lo}, {cur_sel}, {bit_shift};")
                        body.append(f"  @{p_nz} shl.b32 {hi}, {nxt}, {inv_bit};")
                    else:
                        assert isinstance(expr, AShr)
                        body.append(f"  @{p_nz} shr.u32 {lo}, {cur_sel}, {bit_shift};")
                        body.append(f"  @{p_nz} shl.b32 {hi}, {nxt}, {inv_bit};")
                    comb = new_r()
                    body.append(f"  or.b32 {comb}, {lo}, {hi};")
                    body.append(f"  @{p_nz} mov.u32 {res}, {comb};")
                    words.append(res)

            v = emit_mask_value(_V(kind="bits", width=t.width, words=words))
            memo[key] = v
            return v

        if isinstance(expr, Eq):
            a_t = infer_type(expr.a, ctx_types)
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            if isinstance(a_t, BoolType):
                if (
                    a.kind != "pred"
                    or a.pred is None
                    or b.kind != "pred"
                    or b.pred is None
                ):
                    raise CodegenError("bool eq requires pred operands")
                p_x = new_p()
                body.append(f"  xor.pred {p_x}, {a.pred}, {b.pred};")
                p = new_p()
                body.append(f"  not.pred {p}, {p_x};")
                v = _V(kind="pred", pred=p)
                memo[key] = v
                return v

            assert isinstance(a_t, BitVecType)
            if a.kind == "pred" and a.pred is not None:
                a = emit_bool_to_bits(a.pred, a_t.width)
            if b.kind == "pred" and b.pred is not None:
                b = emit_bool_to_bits(b.pred, a_t.width)
            if (
                a.kind != "bits"
                or b.kind != "bits"
                or a.words is None
                or b.words is None
            ):
                raise CodegenError("eq requires bits operands")
            if a.width != a_t.width or b.width != a_t.width:
                raise CodegenError("eq width mismatch")

            p = new_p()
            body.append(f"  mov.pred {p}, 1;")
            n = _words_for_width(a_t.width)
            for i in range(n):
                pi = new_p()
                body.append(f"  setp.eq.u32 {pi}, {a.words[i]}, {b.words[i]};")
                pj = new_p()
                body.append(f"  and.pred {pj}, {p}, {pi};")
                p = pj
            v = _V(kind="pred", pred=p)
            memo[key] = v
            return v

        if isinstance(expr, (Ult, Ule, Ugt, Uge)):
            t = infer_type(expr.a, ctx_types)
            assert isinstance(t, BitVecType)
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            if a.kind == "pred" and a.pred is not None:
                a = emit_bool_to_bits(a.pred, t.width)
            if b.kind == "pred" and b.pred is not None:
                b = emit_bool_to_bits(b.pred, t.width)
            if (
                a.kind != "bits"
                or b.kind != "bits"
                or a.words is None
                or b.words is None
            ):
                raise CodegenError("unsigned compare requires bits operands")
            if a.width != t.width or b.width != t.width:
                raise CodegenError("unsigned compare width mismatch")

            le = isinstance(expr, (Ule, Uge))
            if isinstance(expr, (Ugt, Uge)):
                a, b = b, a

            n = _words_for_width(t.width)
            p_res = new_p()
            body.append(f"  mov.pred {p_res}, 0;")

            lbl_true = new_lbl("UCMP_TRUE")
            lbl_false = new_lbl("UCMP_FALSE")
            lbl_end = new_lbl("UCMP_END")

            for i in range(n - 1, -1, -1):
                p_lt = new_p()
                p_gt = new_p()
                body.append(f"  setp.lt.u32 {p_lt}, {a.words[i]}, {b.words[i]};")
                body.append(f"  setp.gt.u32 {p_gt}, {a.words[i]}, {b.words[i]};")
                body.append(f"  @{p_lt} bra {lbl_true};")
                body.append(f"  @{p_gt} bra {lbl_false};")

            body.append(f"  bra {lbl_true if le else lbl_false};")
            body.append(f"{lbl_true}:")
            body.append(f"  mov.pred {p_res}, 1;")
            body.append(f"  bra {lbl_end};")
            body.append(f"{lbl_false}:")
            body.append(f"  mov.pred {p_res}, 0;")
            body.append(f"{lbl_end}:")

            v = _V(kind="pred", pred=p_res)
            memo[key] = v
            return v

        if isinstance(expr, Mux):
            cond = emit_expr(expr.cond)
            if cond.kind != "pred" or cond.pred is None:
                raise CodegenError("mux condition must be bool")
            out_t = infer_type(expr, ctx_types)
            if isinstance(out_t, BoolType):
                a = emit_expr(expr.a)
                b = emit_expr(expr.b)
                if (
                    a.kind != "pred"
                    or a.pred is None
                    or b.kind != "pred"
                    or b.pred is None
                ):
                    raise CodegenError("bool mux requires pred branches")
                ua = emit_bool_to_bits(a.pred, 32)
                ub = emit_bool_to_bits(b.pred, 32)
                r = new_r()
                body.append(
                    f"  selp.b32 {r}, {ua.words[0]}, {ub.words[0]}, {cond.pred};"
                )
                p = new_p()
                body.append(f"  setp.ne.u32 {p}, {r}, 0;")
                v = _V(kind="pred", pred=p)
                memo[key] = v
                return v

            assert isinstance(out_t, BitVecType)
            a = emit_expr(expr.a)
            b = emit_expr(expr.b)
            if a.kind == "pred" and a.pred is not None:
                a = emit_bool_to_bits(a.pred, out_t.width)
            if b.kind == "pred" and b.pred is not None:
                b = emit_bool_to_bits(b.pred, out_t.width)
            if (
                a.kind != "bits"
                or b.kind != "bits"
                or a.words is None
                or b.words is None
            ):
                raise CodegenError("bitvec mux requires bits branches")
            if a.width != out_t.width or b.width != out_t.width:
                raise CodegenError("mux width mismatch")

            n = _words_for_width(out_t.width)
            out_words: list[str] = []
            for i in range(n):
                r = new_r()
                body.append(f"  selp.b32 {r}, {a.words[i]}, {b.words[i]}, {cond.pred};")
                out_words.append(r)
            v = emit_mask_value(_V(kind="bits", width=out_t.width, words=out_words))
            memo[key] = v
            return v

        if isinstance(expr, Slice):
            src_t = infer_type(expr.x, ctx_types)
            if not isinstance(src_t, BitVecType):
                raise CodegenError("slice requires bitvec source")
            x = emit_expr(expr.x)
            if x.kind == "pred" and x.pred is not None:
                x = emit_bool_to_bits(x.pred, src_t.width)
            if x.kind != "bits" or x.words is None or x.width is None:
                raise CodegenError("slice requires bits source")
            if x.width != src_t.width:
                raise CodegenError("slice width mismatch")

            if expr.width == 1:
                bit_off = expr.offset
                src_word = bit_off // 32
                src_bit = bit_off % 32
                r = new_r()
                body.append(f"  shr.u32 {r}, {x.words[src_word]}, {src_bit};")
                r2 = new_r()
                body.append(f"  and.b32 {r2}, {r}, 1;")
                p = new_p()
                body.append(f"  setp.ne.u32 {p}, {r2}, 0;")
                v = _V(kind="pred", pred=p)
                memo[key] = v
                return v

            out_w = expr.width
            out_n = _words_for_width(out_w)
            out_words: list[str] = []
            for i in range(out_n):
                bit_off = expr.offset + i * 32
                src_word = bit_off // 32
                src_bit = bit_off % 32
                lo = new_r()
                body.append(f"  mov.u32 {lo}, 0;")
                if src_word < len(x.words):
                    body.append(f"  shr.u32 {lo}, {x.words[src_word]}, {src_bit};")
                if src_bit != 0:
                    hi = new_r()
                    body.append(f"  mov.u32 {hi}, 0;")
                    if src_word + 1 < len(x.words):
                        body.append(
                            f"  shl.b32 {hi}, {x.words[src_word + 1]}, {32 - src_bit};"
                        )
                    comb = new_r()
                    body.append(f"  or.b32 {comb}, {lo}, {hi};")
                    out_words.append(comb)
                else:
                    out_words.append(lo)
            v = emit_mask_value(_V(kind="bits", width=out_w, words=out_words))
            memo[key] = v
            return v

        if isinstance(expr, Concat):
            out_t = infer_type(expr, ctx_types)
            assert isinstance(out_t, BitVecType)
            out_n = _words_for_width(out_t.width)
            out_words: list[str] = []
            for _ in range(out_n):
                z = new_r()
                body.append(f"  mov.u32 {z}, 0;")
                out_words.append(z)

            def or_shifted(part: _V, shift_bits: int) -> None:
                nonlocal out_words
                if part.kind == "pred":
                    if part.pred is None:
                        raise CodegenError("concat part pred missing")
                    part = emit_bool_to_bits(part.pred, 1)
                if part.kind != "bits" or part.words is None or part.width is None:
                    raise CodegenError("concat part must be bits/bool")
                k = shift_bits // 32
                r = shift_bits % 32
                for di in range(out_n):
                    si = di - k
                    if si < 0 or si >= _words_for_width(part.width):
                        continue
                    lo = part.words[si]
                    if r == 0:
                        acc = new_r()
                        body.append(f"  or.b32 {acc}, {out_words[di]}, {lo};")
                        out_words[di] = acc
                        continue
                    lo_sh = new_r()
                    body.append(f"  shl.b32 {lo_sh}, {lo}, {r};")
                    hi = None
                    if si - 1 >= 0:
                        hi = part.words[si - 1]
                    hi_sh = new_r()
                    body.append(f"  mov.u32 {hi_sh}, 0;")
                    if hi is not None:
                        body.append(f"  shr.u32 {hi_sh}, {hi}, {32 - r};")
                    comb = new_r()
                    body.append(f"  or.b32 {comb}, {lo_sh}, {hi_sh};")
                    acc = new_r()
                    body.append(f"  or.b32 {acc}, {out_words[di]}, {comb};")
                    out_words[di] = acc

            shift = 0
            for part in reversed(expr.parts):
                pt = infer_type(part, ctx_types)
                pv = emit_expr(part)
                if isinstance(pt, BoolType):
                    or_shifted(pv, shift)
                    shift += 1
                else:
                    assert isinstance(pt, BitVecType)
                    if pv.kind == "pred":
                        raise CodegenError("concat bitvec part must be bits")
                    if pv.kind != "bits" or pv.words is None or pv.width is None:
                        raise CodegenError("concat bitvec part must be bits")
                    if pv.width != pt.width:
                        raise CodegenError("concat part width mismatch")
                    pv = emit_mask_value(pv)
                    or_shifted(pv, shift)
                    shift += pt.width
            v = emit_mask_value(_V(kind="bits", width=out_t.width, words=out_words))
            memo[key] = v
            return v

        raise CodegenError("unsupported expression for PTX backend")

    rd_in = new_rd()
    rd_state_in = new_rd() if ir.state else None
    rd_out = new_rd()
    rd_state_out = new_rd() if ir.state else None
    r_n = new_r()
    r_tid = new_r()
    r_cta = new_r()
    r_ntid = new_r()
    r_idx = new_r()
    p_oob = new_p()

    pre: list[str] = []
    pre.append(f"  ld.param.u64 {rd_in}, [__in];")
    if ir.state:
        assert rd_state_in is not None and rd_state_out is not None
        pre.append(f"  ld.param.u64 {rd_state_in}, [__state_in];")
    pre.append(f"  ld.param.u64 {rd_out}, [__out];")
    if ir.state:
        assert rd_state_out is not None
        pre.append(f"  ld.param.u64 {rd_state_out}, [__state_out];")
    pre.append(f"  ld.param.u32 {r_n}, [__n];")
    pre.append(f"  mov.u32 {r_tid}, %tid.x;")
    pre.append(f"  mov.u32 {r_cta}, %ctaid.x;")
    pre.append(f"  mov.u32 {r_ntid}, %ntid.x;")
    pre.append(f"  mad.lo.u32 {r_idx}, {r_cta}, {r_ntid}, {r_tid};")
    pre.append(f"  setp.ge.u32 {p_oob}, {r_idx}, {r_n};")
    pre.append(f"  @{p_oob} bra DONE;")

    body = pre + body

    for name in input_order:
        env[name] = emit_load_port(
            ir.inputs[name], r_idx, rd_in, in_stride, in_offs[name]
        )

    if ir.state:
        assert rd_state_in is not None
        if assume_reset_state:
            for name in runtime_state_order:
                t = ir.state[name]
                v = reset_vals_all[name]
                if isinstance(t, BoolType):
                    env[name] = emit_expr(BoolConst(value=bool(v)))
                else:
                    assert isinstance(t, BitVecType)
                    env[name] = emit_expr(BitVecConst(width=t.width, value=int(v)))
        else:
            for name in state_order:
                env[name] = emit_load_port(
                    ir.state[name], r_idx, rd_state_in, st_stride, st_offs[name]
                )

    use_runtime_loop = (
        bool(runtime_step_loop)
        and int(steps) > 1
        and bool(ir.state)
        and schedule is None
    )

    last_outputs: dict[str, _V] = {}
    if use_runtime_loop:
        r_step = new_r()
        p_done = new_p()
        loop_head = new_lbl("STEP_LOOP")
        loop_done = new_lbl("STEP_DONE")
        body.append(f"  mov.u32 {r_step}, 0;")
        body.append(f"{loop_head}:")
        body.append(f"  setp.ge.u32 {p_done}, {r_step}, {int(steps)};")
        body.append(f"  @{p_done} bra {loop_done};")

        memo = {}
        step_outputs: dict[str, _V] = {}
        for name in output_order:
            step_outputs[name] = emit_expr(ir.output_exprs[name])
        last_outputs = step_outputs

        next_env: dict[str, _V] = {}
        for name in runtime_state_order:
            next_env[name] = emit_expr(ir.next_state[name])
        for name in runtime_state_order:
            emit_copy_value(env[name], next_env[name], ir.state[name])

        body.append(f"  add.u32 {r_step}, {r_step}, 1;")
        body.append(f"  bra {loop_head};")
        body.append(f"{loop_done}:")
    else:
        for step_index in range(int(steps)):
            memo = {}
            repl = (
                const_state_repl_for_step(ir, schedule, step_index=step_index)
                if schedule is not None and const_state_steps
                else {}
            )
            step_outputs: dict[str, _V] = {}
            for name in output_order:
                expr = specialize_expr(ir.output_exprs[name], ctx_types, repl)
                step_outputs[name] = emit_expr(expr)
            last_outputs = step_outputs

            if ir.state:
                next_env: dict[str, _V] = {}
                for name in runtime_state_order:
                    expr = specialize_expr(ir.next_state[name], ctx_types, repl)
                    next_env[name] = emit_expr(expr)
                for name in runtime_state_order:
                    env[name] = next_env[name]

    for name in output_order:
        v = last_outputs[name]
        emit_store_port(v, ir.outputs[name], r_idx, rd_out, out_stride, out_offs[name])

    if ir.state:
        assert rd_state_out is not None
        for name in runtime_state_order:
            v = env[name]
            emit_store_port(
                v, ir.state[name], r_idx, rd_state_out, st_stride, st_offs[name]
            )
        if const_state_order:
            memo = {}
            for name in const_state_order:
                t = ir.state[name]
                v = const_state_final[name]
                if isinstance(t, BoolType):
                    vv = emit_expr(BoolConst(value=bool(v)))
                else:
                    assert isinstance(t, BitVecType)
                    vv = emit_expr(BitVecConst(width=t.width, value=int(v)))
                emit_store_port(
                    vv, ir.state[name], r_idx, rd_state_out, st_stride, st_offs[name]
                )

    body.append("DONE:")
    body.append("  ret;")

    lines.append(".version 6.4")
    lines.append(f".target {sm}")
    lines.append(".address_size 64")
    lines.append("")
    if lut_decls:
        lines.extend(lut_decls)
    lines.append(".visible .entry stc_eval(")
    if ir.state:
        lines.append("  .param .u64 __in,")
        lines.append("  .param .u64 __state_in,")
        lines.append("  .param .u64 __out,")
        lines.append("  .param .u64 __state_out,")
        lines.append("  .param .u32 __n")
    else:
        lines.append("  .param .u64 __in,")
        lines.append("  .param .u64 __out,")
        lines.append("  .param .u32 __n")
    lines.append(")")
    lines.append("{")
    if p_cnt:
        lines.append(f"  .reg .pred %p<{p_cnt}>;")
    if r_cnt:
        lines.append(f"  .reg .u32 %r<{r_cnt}>;")
    if rd_cnt:
        lines.append(f"  .reg .u64 %rd<{rd_cnt}>;")
    lines.extend(body)
    lines.append("}")
    lines.append("")
    return "\n".join(lines)
