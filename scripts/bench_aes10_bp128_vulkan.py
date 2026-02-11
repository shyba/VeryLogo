#!/usr/bin/env python3
"""
Benchmark AES-128 (10 rounds) on Vulkan using a BP128-style S-box subroutine.

Kernel shape:
- One invocation processes 32 AES blocks in bit-sliced form (one bit-plane per u32 word)
- S-box is a callable SPIR-V function generated from BP128 xor/and gates
- ShiftRows/MixColumns/AddRoundKey are emitted manually around the S-box calls

This intentionally mirrors the CUDA "BP128 subroutine + manual orchestration" experiment
for Vulkan backends.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bp_circuit_sbox import build_bp_sbox


ROUND_KEYS_HEX = [
    "000102030405060708090a0b0c0d0e0f",
    "d6aa74fdd2af72fadaa678f1d6ab76fe",
    "b692cf0b643dbdf1be9bc5006830b3fe",
    "b6ff744ed2c2c9bf6c590cbf0469bf41",
    "47f7f7bc95353e03f96c32bcfd058dfd",
    "3caaa3e8a99f9deb50f3af57adf622aa",
    "5e390f7df7a69296a7553dc10aa31f6b",
    "14f9701ae35fe28c440adf4d4ea9c026",
    "47438735a41c65b9e016baf4aebf7ad2",
    "549932d1f08557681093ed9cbe2c974e",
    "13111d7fe3944a17f307a78b4d2b30c5",
]

AES_SBOX = [
    0x63,
    0x7C,
    0x77,
    0x7B,
    0xF2,
    0x6B,
    0x6F,
    0xC5,
    0x30,
    0x01,
    0x67,
    0x2B,
    0xFE,
    0xD7,
    0xAB,
    0x76,
    0xCA,
    0x82,
    0xC9,
    0x7D,
    0xFA,
    0x59,
    0x47,
    0xF0,
    0xAD,
    0xD4,
    0xA2,
    0xAF,
    0x9C,
    0xA4,
    0x72,
    0xC0,
    0xB7,
    0xFD,
    0x93,
    0x26,
    0x36,
    0x3F,
    0xF7,
    0xCC,
    0x34,
    0xA5,
    0xE5,
    0xF1,
    0x71,
    0xD8,
    0x31,
    0x15,
    0x04,
    0xC7,
    0x23,
    0xC3,
    0x18,
    0x96,
    0x05,
    0x9A,
    0x07,
    0x12,
    0x80,
    0xE2,
    0xEB,
    0x27,
    0xB2,
    0x75,
    0x09,
    0x83,
    0x2C,
    0x1A,
    0x1B,
    0x6E,
    0x5A,
    0xA0,
    0x52,
    0x3B,
    0xD6,
    0xB3,
    0x29,
    0xE3,
    0x2F,
    0x84,
    0x53,
    0xD1,
    0x00,
    0xED,
    0x20,
    0xFC,
    0xB1,
    0x5B,
    0x6A,
    0xCB,
    0xBE,
    0x39,
    0x4A,
    0x4C,
    0x58,
    0xCF,
    0xD0,
    0xEF,
    0xAA,
    0xFB,
    0x43,
    0x4D,
    0x33,
    0x85,
    0x45,
    0xF9,
    0x02,
    0x7F,
    0x50,
    0x3C,
    0x9F,
    0xA8,
    0x51,
    0xA3,
    0x40,
    0x8F,
    0x92,
    0x9D,
    0x38,
    0xF5,
    0xBC,
    0xB6,
    0xDA,
    0x21,
    0x10,
    0xFF,
    0xF3,
    0xD2,
    0xCD,
    0x0C,
    0x13,
    0xEC,
    0x5F,
    0x97,
    0x44,
    0x17,
    0xC4,
    0xA7,
    0x7E,
    0x3D,
    0x64,
    0x5D,
    0x19,
    0x73,
    0x60,
    0x81,
    0x4F,
    0xDC,
    0x22,
    0x2A,
    0x90,
    0x88,
    0x46,
    0xEE,
    0xB8,
    0x14,
    0xDE,
    0x5E,
    0x0B,
    0xDB,
    0xE0,
    0x32,
    0x3A,
    0x0A,
    0x49,
    0x06,
    0x24,
    0x5C,
    0xC2,
    0xD3,
    0xAC,
    0x62,
    0x91,
    0x95,
    0xE4,
    0x79,
    0xE7,
    0xC8,
    0x37,
    0x6D,
    0x8D,
    0xD5,
    0x4E,
    0xA9,
    0x6C,
    0x56,
    0xF4,
    0xEA,
    0x65,
    0x7A,
    0xAE,
    0x08,
    0xBA,
    0x78,
    0x25,
    0x2E,
    0x1C,
    0xA6,
    0xB4,
    0xC6,
    0xE8,
    0xDD,
    0x74,
    0x1F,
    0x4B,
    0xBD,
    0x8B,
    0x8A,
    0x70,
    0x3E,
    0xB5,
    0x66,
    0x48,
    0x03,
    0xF6,
    0x0E,
    0x61,
    0x35,
    0x57,
    0xB9,
    0x86,
    0xC1,
    0x1D,
    0x9E,
    0xE1,
    0xF8,
    0x98,
    0x11,
    0x69,
    0xD9,
    0x8E,
    0x94,
    0x9B,
    0x1E,
    0x87,
    0xE9,
    0xCE,
    0x55,
    0x28,
    0xDF,
    0x8C,
    0xA1,
    0x89,
    0x0D,
    0xBF,
    0xE6,
    0x42,
    0x68,
    0x41,
    0x99,
    0x2D,
    0x0F,
    0xB0,
    0x54,
    0xBB,
    0x16,
]


def _aes_xtime(x: int) -> int:
    x &= 0xFF
    return ((x << 1) & 0xFF) ^ (0x1B if (x & 0x80) else 0)


def _aes_round_ref(state: bytes, rk: bytes) -> bytes:
    sb = bytes(AES_SBOX[b] for b in state)
    sr = bytes(
        [
            sb[0],
            sb[5],
            sb[10],
            sb[15],
            sb[4],
            sb[9],
            sb[14],
            sb[3],
            sb[8],
            sb[13],
            sb[2],
            sb[7],
            sb[12],
            sb[1],
            sb[6],
            sb[11],
        ]
    )
    mc = [0] * 16
    for c in range(4):
        i0 = c * 4 + 0
        i1 = c * 4 + 1
        i2 = c * 4 + 2
        i3 = c * 4 + 3
        a0, a1, a2, a3 = sr[i0], sr[i1], sr[i2], sr[i3]
        t = a0 ^ a1 ^ a2 ^ a3
        u = a0
        mc[i0] = a0 ^ t ^ _aes_xtime(a0 ^ a1)
        mc[i1] = a1 ^ t ^ _aes_xtime(a1 ^ a2)
        mc[i2] = a2 ^ t ^ _aes_xtime(a2 ^ a3)
        mc[i3] = a3 ^ t ^ _aes_xtime(a3 ^ u)
    return bytes((mc[i] ^ rk[i]) & 0xFF for i in range(16))


def _aes_encrypt_10round_ref(pt: bytes, round_keys: list[bytes]) -> bytes:
    st = bytes((pt[i] ^ round_keys[0][i]) & 0xFF for i in range(16))
    for r in range(1, 10):
        st = _aes_round_ref(st, round_keys[r])
    sb = bytes(AES_SBOX[b] for b in st)
    sr = bytes(
        [
            sb[0],
            sb[5],
            sb[10],
            sb[15],
            sb[4],
            sb[9],
            sb[14],
            sb[3],
            sb[8],
            sb[13],
            sb[2],
            sb[7],
            sb[12],
            sb[1],
            sb[6],
            sb[11],
        ]
    )
    return bytes((sr[i] ^ round_keys[10][i]) & 0xFF for i in range(16))


class SpvBuilder:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.id_counter = 1
        self.const_u32: dict[int, str] = {}

    def nid(self) -> str:
        out = f"%{self.id_counter}"
        self.id_counter += 1
        return out

    def add(self, text: str) -> None:
        self.lines.append(text)

    def add_blank(self) -> None:
        self.lines.append("")

    def u32(self, value: int, uint_id: str) -> str:
        value &= 0xFFFFFFFF
        if value not in self.const_u32:
            cid = self.nid()
            self.add(f"{cid} = OpConstant {uint_id} {value}")
            self.const_u32[value] = cid
        return self.const_u32[value]

    def finish(self) -> str:
        return "\n".join(self.lines) + "\n"


def _bit_index(byte_idx: int, bit_idx: int) -> int:
    return byte_idx * 8 + bit_idx


def _rk_bit(round_idx: int, byte_idx: int, bit_idx: int) -> int:
    b = bytes.fromhex(ROUND_KEYS_HEX[round_idx])[byte_idx]
    return (b >> bit_idx) & 1


def _emit_spirv_aes10_bp128(
    local_size: tuple[int, int, int] = (64, 1, 1),
    *,
    input_words: int = 128,
    output_words: int = 128,
) -> str:
    bp = build_bp_sbox()
    sb = SpvBuilder()

    sb.add("; SPIR-V")
    sb.add("; Version: 1.3")
    sb.add("; Generator: VeryLogo BP128 AES10 Vulkan")
    sb.add("; Bound: 100000")
    sb.add("; Schema: 0")
    sb.add_blank()
    sb.add("OpCapability Shader")
    sb.add('OpExtension "SPV_KHR_storage_buffer_storage_class"')
    sb.add("OpMemoryModel Logical GLSL450")

    main_id = sb.nid()
    gid_var_id = sb.nid()
    sb.add(f'OpEntryPoint GLCompute {main_id} "main" {gid_var_id}')
    sb.add(
        f"OpExecutionMode {main_id} LocalSize {local_size[0]} {local_size[1]} {local_size[2]}"
    )
    sb.add_blank()

    arr_id = sb.nid()
    in_struct_id = sb.nid()
    out_struct_id = sb.nid()
    in_buf_id = sb.nid()
    out_buf_id = sb.nid()

    sb.add(f"OpDecorate {gid_var_id} BuiltIn GlobalInvocationId")
    sb.add(f"OpDecorate {arr_id} ArrayStride 4")
    sb.add(f"OpMemberDecorate {in_struct_id} 0 Offset 0")
    sb.add(f"OpDecorate {in_struct_id} Block")
    sb.add(f"OpDecorate {in_buf_id} DescriptorSet 0")
    sb.add(f"OpDecorate {in_buf_id} Binding 0")
    sb.add(f"OpMemberDecorate {out_struct_id} 0 Offset 0")
    sb.add(f"OpDecorate {out_struct_id} Block")
    sb.add(f"OpDecorate {out_buf_id} DescriptorSet 0")
    sb.add(f"OpDecorate {out_buf_id} Binding 1")
    sb.add_blank()

    void_id = sb.nid()
    uint_id = sb.nid()
    int_id = sb.nid()
    v3uint_id = sb.nid()
    ptr_input_v3uint = sb.nid()
    ptr_storage_uint = sb.nid()
    ptr_storage_in = sb.nid()
    ptr_storage_out = sb.nid()
    main_func_type = sb.nid()

    sb.add(f"{void_id} = OpTypeVoid")
    sb.add(f"{uint_id} = OpTypeInt 32 0")
    sb.add(f"{int_id} = OpTypeInt 32 1")
    sb.add(f"{v3uint_id} = OpTypeVector {uint_id} 3")
    sb.add(f"{ptr_input_v3uint} = OpTypePointer Input {v3uint_id}")
    sb.add(f"{ptr_storage_uint} = OpTypePointer StorageBuffer {uint_id}")
    sb.add(f"{arr_id} = OpTypeRuntimeArray {uint_id}")
    sb.add(f"{in_struct_id} = OpTypeStruct {arr_id}")
    sb.add(f"{out_struct_id} = OpTypeStruct {arr_id}")
    sb.add(f"{ptr_storage_in} = OpTypePointer StorageBuffer {in_struct_id}")
    sb.add(f"{ptr_storage_out} = OpTypePointer StorageBuffer {out_struct_id}")
    sb.add(f"{main_func_type} = OpTypeFunction {void_id}")

    sbox_ret_type = sb.nid()
    sb.add(f"{sbox_ret_type} = OpTypeStruct {' '.join([uint_id] * 8)}")
    sbox_param_types = [uint_id] * 8
    sbox_func_type = sb.nid()
    sb.add(f"{sbox_func_type} = OpTypeFunction {sbox_ret_type} {' '.join(sbox_param_types)}")
    sb.add_blank()

    int_0 = sb.nid()
    sb.add(f"{int_0} = OpConstant {int_id} 0")
    c0 = sb.u32(0, uint_id)
    c_ffff = sb.u32(0xFFFFFFFF, uint_id)
    c_input_words = sb.u32(input_words, uint_id)
    _ = sb.u32(output_words, uint_id)
    for i in range(max(input_words, output_words)):
        _ = sb.u32(i, uint_id)
    sb.add_blank()

    sb.add(f"{gid_var_id} = OpVariable {ptr_input_v3uint} Input")
    sb.add(f"{in_buf_id} = OpVariable {ptr_storage_in} StorageBuffer")
    sb.add(f"{out_buf_id} = OpVariable {ptr_storage_out} StorageBuffer")
    sb.add_blank()

    # BP128 S-box function.
    sbox_fn_id = sb.nid()
    sb.add(f"{sbox_fn_id} = OpFunction {sbox_ret_type} None {sbox_func_type}")
    in_params = [sb.nid() for _ in range(8)]
    for pid in in_params:
        sb.add(f"{pid} = OpFunctionParameter {uint_id}")
    sbox_entry = sb.nid()
    sb.add(f"{sbox_entry} = OpLabel")

    reg_map: dict[int, str] = {i: in_params[i] for i in range(8)}
    for gate_index, gate in enumerate(bp.gates):
        dst_node = bp.input_bits + gate_index
        op = gate[0]
        if op == "xor":
            a, b = int(gate[1]), int(gate[2])
            dst = sb.nid()
            sb.add(f"{dst} = OpBitwiseXor {uint_id} {reg_map[a]} {reg_map[b]}")
            reg_map[dst_node] = dst
        elif op == "and":
            a, b = int(gate[1]), int(gate[2])
            dst = sb.nid()
            sb.add(f"{dst} = OpBitwiseAnd {uint_id} {reg_map[a]} {reg_map[b]}")
            reg_map[dst_node] = dst
        else:
            raise RuntimeError(f"Unsupported BP128 gate op in S-box function: {op}")

    sbox_out_vals: list[str] = []
    for out_idx, (node, inv) in enumerate(bp.outputs):
        val = reg_map[int(node)]
        if inv:
            not_id = sb.nid()
            sb.add(f"{not_id} = OpNot {uint_id} {val}")
            val = not_id
        sbox_out_vals.append(val)
    ret_val = sb.nid()
    sb.add(f"{ret_val} = OpCompositeConstruct {sbox_ret_type} {' '.join(sbox_out_vals)}")
    sb.add(f"OpReturnValue {ret_val}")
    sb.add("OpFunctionEnd")
    sb.add_blank()

    # Main function.
    sb.add(f"{main_id} = OpFunction {void_id} None {main_func_type}")
    main_entry = sb.nid()
    sb.add(f"{main_entry} = OpLabel")

    gid = sb.nid()
    tid = sb.nid()
    base = sb.nid()
    sb.add(f"{gid} = OpLoad {v3uint_id} {gid_var_id}")
    sb.add(f"{tid} = OpCompositeExtract {uint_id} {gid} 0")
    sb.add(f"{base} = OpIMul {uint_id} {tid} {c_input_words}")

    def emit_xor(a: str, b: str) -> str:
        if a == c0:
            return b
        if b == c0:
            return a
        if a == b:
            return c0
        dst = sb.nid()
        sb.add(f"{dst} = OpBitwiseXor {uint_id} {a} {b}")
        return dst

    def emit_load_word(word_index: int) -> str:
        off = sb.nid()
        ptr = sb.nid()
        val = sb.nid()
        sb.add(f"{off} = OpIAdd {uint_id} {base} {sb.const_u32[word_index]}")
        sb.add(f"{ptr} = OpAccessChain {ptr_storage_uint} {in_buf_id} {int_0} {off}")
        sb.add(f"{val} = OpLoad {uint_id} {ptr}")
        return val

    def emit_store_word(word_index: int, value: str) -> None:
        off = sb.nid()
        ptr = sb.nid()
        sb.add(f"{off} = OpIAdd {uint_id} {base} {sb.const_u32[word_index]}")
        sb.add(f"{ptr} = OpAccessChain {ptr_storage_uint} {out_buf_id} {int_0} {off}")
        sb.add(f"OpStore {ptr} {value}")

    def xtime(bits: list[str]) -> list[str]:
        b7 = bits[7]
        return [
            b7,
            emit_xor(bits[0], b7),
            bits[1],
            emit_xor(bits[2], b7),
            emit_xor(bits[3], b7),
            bits[4],
            bits[5],
            bits[6],
        ]

    state = [emit_load_word(i) for i in range(128)]

    # Initial AddRoundKey.
    for byte in range(16):
        for bit in range(8):
            if _rk_bit(0, byte, bit):
                idx = _bit_index(byte, bit)
                state[idx] = emit_xor(state[idx], c_ffff)

    def sub_bytes(cur_state: list[str]) -> list[str]:
        out = [c0] * 128
        for byte in range(16):
            ins = [cur_state[_bit_index(byte, bit)] for bit in range(8)]
            call_res = sb.nid()
            sb.add(
                f"{call_res} = OpFunctionCall {sbox_ret_type} {sbox_fn_id} {' '.join(ins)}"
            )
            for bit in range(8):
                val = sb.nid()
                sb.add(f"{val} = OpCompositeExtract {uint_id} {call_res} {bit}")
                out[_bit_index(byte, bit)] = val
        return out

    def shift_rows(cur_state: list[str]) -> list[str]:
        out = [c0] * 128
        perm = [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11]
        for dst_byte, src_byte in enumerate(perm):
            for bit in range(8):
                out[_bit_index(dst_byte, bit)] = cur_state[_bit_index(src_byte, bit)]
        return out

    def mix_columns(cur_state: list[str]) -> list[str]:
        out = [c0] * 128
        for c in range(4):
            i0 = c * 4 + 0
            i1 = c * 4 + 1
            i2 = c * 4 + 2
            i3 = c * 4 + 3
            a0 = [cur_state[_bit_index(i0, b)] for b in range(8)]
            a1 = [cur_state[_bit_index(i1, b)] for b in range(8)]
            a2 = [cur_state[_bit_index(i2, b)] for b in range(8)]
            a3 = [cur_state[_bit_index(i3, b)] for b in range(8)]
            t = [emit_xor(emit_xor(a0[b], a1[b]), emit_xor(a2[b], a3[b])) for b in range(8)]
            x01 = [emit_xor(a0[b], a1[b]) for b in range(8)]
            x12 = [emit_xor(a1[b], a2[b]) for b in range(8)]
            x23 = [emit_xor(a2[b], a3[b]) for b in range(8)]
            x30 = [emit_xor(a3[b], a0[b]) for b in range(8)]
            xt01 = xtime(x01)
            xt12 = xtime(x12)
            xt23 = xtime(x23)
            xt30 = xtime(x30)
            for b in range(8):
                out[_bit_index(i0, b)] = emit_xor(emit_xor(a0[b], t[b]), xt01[b])
                out[_bit_index(i1, b)] = emit_xor(emit_xor(a1[b], t[b]), xt12[b])
                out[_bit_index(i2, b)] = emit_xor(emit_xor(a2[b], t[b]), xt23[b])
                out[_bit_index(i3, b)] = emit_xor(emit_xor(a3[b], t[b]), xt30[b])
        return out

    # Rounds 1..9.
    for round_idx in range(1, 10):
        state = sub_bytes(state)
        state = shift_rows(state)
        state = mix_columns(state)
        for byte in range(16):
            for bit in range(8):
                if _rk_bit(round_idx, byte, bit):
                    idx = _bit_index(byte, bit)
                    state[idx] = emit_xor(state[idx], c_ffff)

    # Final round (no MixColumns).
    state = sub_bytes(state)
    state = shift_rows(state)
    for byte in range(16):
        for bit in range(8):
            if _rk_bit(10, byte, bit):
                idx = _bit_index(byte, bit)
                state[idx] = emit_xor(state[idx], c_ffff)

    for i in range(128):
        emit_store_word(i, state[i])

    sb.add("OpReturn")
    sb.add("OpFunctionEnd")

    out = sb.finish()
    out = out.replace("; Bound: 100000", f"; Bound: {sb.id_counter + 32}")
    return out


def _parse_dump_words(stdout: str) -> dict[int, int]:
    words: dict[int, int] = {}
    in_section = False
    for line in stdout.splitlines():
        if line.startswith("DUMP_"):
            if in_section:
                break
            in_section = True
            continue
        if not in_section:
            continue
        m = re.search(r"^\s*\[(\d+)\]=0x([0-9a-fA-F]{8})\s*$", line)
        if m:
            words[int(m.group(1))] = int(m.group(2), 16)
    return words


def _expected_ciphertexts_for_thread0() -> list[bytes]:
    round_keys = [bytes.fromhex(x) for x in ROUND_KEYS_HEX]
    out: list[bytes] = []
    for lane in range(32):
        pt = bytearray(16)
        for byte_idx in range(16):
            value = 0
            for bit in range(8):
                word_idx = _bit_index(byte_idx, bit)
                word = (0xA5A50000 + word_idx) & 0xFFFFFFFF
                value |= ((word >> lane) & 1) << bit
            pt[byte_idx] = value
        out.append(_aes_encrypt_10round_ref(bytes(pt), round_keys))
    return out


def _validate_dump(stdout: str) -> tuple[bool, str]:
    words = _parse_dump_words(stdout)
    if len(words) < 128:
        return False, f"expected 128 dumped words, got {len(words)}"
    expected = _expected_ciphertexts_for_thread0()
    mismatches = 0
    first = ""
    for lane in range(32):
        for byte_idx in range(16):
            exp_byte = expected[lane][byte_idx]
            for bit in range(8):
                idx = _bit_index(byte_idx, bit)
                got = (words[idx] >> lane) & 1
                exp = (exp_byte >> bit) & 1
                if got != exp:
                    mismatches += 1
                    if not first:
                        first = (
                            f"lane={lane} byte={byte_idx} bit={bit} exp={exp} got={got} "
                            f"word=0x{words[idx]:08x}"
                        )
    if mismatches:
        return False, f"{mismatches} mismatches ({first})"
    return True, "lane-wise AES-128 correctness OK for dumped thread0 output"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--threads", type=int, default=32768)
    ap.add_argument("--local-x", type=int, default=64)
    ap.add_argument("--local-y", type=int, default=1)
    ap.add_argument("--local-z", type=int, default=1)
    ap.add_argument("--check", action="store_true", help="Validate output from one dispatch")
    ap.add_argument("--out", default="out/aes10_bp128_vulkan")
    args = ap.parse_args()

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    spvasm = _emit_spirv_aes10_bp128(
        local_size=(args.local_x, args.local_y, args.local_z),
        input_words=128,
        output_words=128,
    )
    spvasm_path = out_prefix.with_suffix(".spvasm")
    spv_path = out_prefix.with_suffix(".spv")
    spvasm_path.write_text(spvasm, encoding="utf-8")
    subprocess.run(["spirv-as", str(spvasm_path), "-o", str(spv_path)], check=True)

    with tempfile.TemporaryDirectory(prefix="aes10_bp128_vk_") as td:
        td_path = Path(td)
        bench_bin = td_path / "bench_vulkan"
        subprocess.run(
            [
                "g++",
                "-O3",
                "-std=c++17",
                "-I.",
                "-o",
                str(bench_bin),
                "scripts/bench_circuit_vulkan.cpp",
                "-lvulkan",
            ],
            check=True,
        )

        check_iters = 1 if args.check else args.iters
        # In check mode we still use host-side check plumbing only to force dump output.
        # width_bits=0 below disables its rk/aes semantic checks.
        check_flag = 1 if args.check else 0
        dump_words = 128 if args.check else 0
        cmd = [
            str(bench_bin),
            str(spv_path),
            str(check_iters),
            str(args.threads),
            str(args.local_x),
            str(args.local_y),
            str(args.local_z),
            "128",  # input words
            "128",  # output words
            "1",  # host rounds
            "32",  # lanes
            "16",  # bytes per eval
            str(check_flag),
            "0",
            "0",
            "0",
            "1",
            str(dump_words),
            "0",
            "1",
            "0",
            "-1",
            "-1",
        ]
        run = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if run.stdout:
            print(run.stdout.strip())
        if run.returncode != 0:
            if run.stderr:
                print(run.stderr.strip())
            return run.returncode

        if args.check:
            ok, msg = _validate_dump(run.stdout)
            if ok:
                print(f"CHECK: {msg}")
            else:
                print(f"CHECK FAILED: {msg}")
                return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
