#!/usr/bin/env python3
import ctypes
import subprocess
from pathlib import Path

from stc.layout_bin import read_packed_layout_bin

Vec = ctypes.c_uint64 * 8

def _aligned_vec_array(n):
    raw = ctypes.create_string_buffer(n * ctypes.sizeof(Vec) + 63)
    base = ctypes.addressof(raw)
    aligned = (base + 63) & ~63
    ptr = ctypes.cast(aligned, ctypes.POINTER(Vec))
    return raw, ptr

ONES = Vec(*([0xFFFFFFFFFFFFFFFF] * 8))
ZEROS = Vec(*([0] * 8))

def _set_field_bits_bitsliced(arr, *, lsb, width, value):
    for i in range(width):
        bit = (value >> i) & 1
        arr[lsb + i] = ONES if bit else ZEROS

def _pack_word_be(bs):
    v = 0
    for i, b in enumerate(bs):
        v |= int(b) << (24 - 8 * i)
    return v

def _out_bits_to_bytes_le(bits_lsb_first):
    assert len(bits_lsb_first) % 8 == 0
    out = bytearray(len(bits_lsb_first) // 8)
    for i in range(len(out)):
        v = 0
        for b in range(8):
            v |= (bits_lsb_first[i * 8 + b] & 1) << b
        out[i] = v
    return bytes(out)

def _out_bits_to_digest_bytes(bits_512_lsb_first):
    raw = _out_bits_to_bytes_le(bits_512_lsb_first)
    if len(raw) != 64:
        raise ValueError("expected 512 bits")
    return raw[::-1]

layout_path = Path("out/keccak_test_no_sched/io_layout.bin")
so_path = Path("out/keccak_test_no_sched/circuit_avx512.so")

layout = read_packed_layout_bin(layout_path).to_dict()
lib = ctypes.CDLL(str(so_path))

inputs = layout["inputs"]
state = layout["state"]
outputs = layout["outputs"]

input_bits = sum(int(v["width"]) for v in inputs.values())
state_bits = sum(int(v["width"]) for v in state.values())
out_bits = sum(int(v["width"]) for v in outputs.values())

in_raw, in_ptr = _aligned_vec_array(input_bits)
out_raw, out_ptr = _aligned_vec_array(out_bits)
st0_raw, st0_ptr = _aligned_vec_array(state_bits)
st1_raw, st1_ptr = _aligned_vec_array(state_bits)

in_arr = ctypes.cast(in_ptr, ctypes.POINTER(Vec * input_bits)).contents
out_arr = ctypes.cast(out_ptr, ctypes.POINTER(Vec * out_bits)).contents
st0_arr = ctypes.cast(st0_ptr, ctypes.POINTER(Vec * state_bits)).contents
st1_arr = ctypes.cast(st1_ptr, ctypes.POINTER(Vec * state_bits)).contents

for i in range(state_bits):
    st0_arr[i] = ZEROS
    st1_arr[i] = ZEROS

lib.circuit_steps_shared.argtypes = [
    ctypes.POINTER(Vec),
    ctypes.POINTER(Vec),
    ctypes.POINTER(Vec),
    ctypes.POINTER(Vec),
    ctypes.c_int,
]
lib.circuit_steps_shared.restype = None

st_ptrs = [st0_ptr, st1_ptr]

def step_once(*, in_word, in_ready_v, is_last_v, byte_num_v, reset_v):
    for i in range(input_bits):
        in_arr[i] = ZEROS
    _set_field_bits_bitsliced(in_arr, lsb=int(inputs["in"]["lsb"]), width=32, value=in_word)
    _set_field_bits_bitsliced(in_arr, lsb=int(inputs["in_ready"]["lsb"]), width=int(inputs["in_ready"]["width"]), value=in_ready_v)
    _set_field_bits_bitsliced(in_arr, lsb=int(inputs["is_last"]["lsb"]), width=int(inputs["is_last"]["width"]), value=is_last_v)
    _set_field_bits_bitsliced(in_arr, lsb=int(inputs["byte_num"]["lsb"]), width=int(inputs["byte_num"]["width"]), value=byte_num_v)
    _set_field_bits_bitsliced(in_arr, lsb=int(inputs["reset"]["lsb"]), width=int(inputs["reset"]["width"]), value=reset_v)
    lib.circuit_steps_shared(in_ptr, out_ptr, st_ptrs[0], st_ptrs[1], 1)
    st_ptrs[0], st_ptrs[1] = st_ptrs[1], st_ptrs[0]

# Reset pulse
step_once(in_word=0, in_ready_v=0, is_last_v=0, byte_num_v=0, reset_v=1)
step_once(in_word=0, in_ready_v=0, is_last_v=0, byte_num_v=0, reset_v=0)

# Feed "abc" message
msg = b"abc"
msg_words = [msg[i:i+4] for i in range(0, len(msg), 4)]
if len(msg_words[-1]) < 4:
    msg_words[-1] += b'\x00' * (4 - len(msg_words[-1]))

for word_idx, word_bytes in enumerate(msg_words):
    is_last = 1 if word_idx == len(msg_words) - 1 else 0
    word = _pack_word_be(word_bytes)
    step_once(in_word=word, in_ready_v=1, is_last_v=is_last, byte_num_v=len(msg), reset_v=0)

# Run idle cycles until digest is ready
out_ready = outputs["out_ready"]
out_word = outputs["out"]
max_cycles = 200
for cycle in range(max_cycles):
    step_once(in_word=0, in_ready_v=0, is_last_v=0, byte_num_v=0, reset_v=0)
    ready_bit = 1 if (int(out_arr[int(out_ready["lsb"])][0]) & 1) else 0
    if ready_bit:
        out_lsb = int(out_word["lsb"])
        out_w = int(out_word["width"])
        bits = [1 if (int(out_arr[out_lsb + i][0]) & 1) else 0 for i in range(out_w)]
        digest = _out_bits_to_digest_bytes(bits)
        print(f"Digest produced after {cycle} idle cycles:")
        print(digest.hex())

        # Compare with openssl
        p = subprocess.run(["openssl", "dgst", "-keccak-512"], input=msg, stdout=subprocess.PIPE, check=True)
        expected = bytes.fromhex(p.stdout.decode().strip().split()[-1])
        if digest == expected:
            print("MATCH with OpenSSL!")
        else:
            print(f"MISMATCH! Expected: {expected.hex()}")
        break
else:
    print(f"ERROR: Digest not produced after {max_cycles} cycles")
