#!/usr/bin/env python3
import ctypes
import json
from pathlib import Path

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

layout_path = Path("out/keccak_test_no_sched/io_layout.json")
so_path = Path("out/keccak_test_no_sched/circuit_avx512.so")

layout = json.loads(layout_path.read_text())
lib = ctypes.CDLL(str(so_path))

inputs = layout["inputs"]
state = layout["state"]
outputs = layout["outputs"]

input_bits = sum(int(v["width"]) for v in inputs.values())
state_bits = sum(int(v["width"]) for v in state.values())
out_bits = sum(int(v["width"]) for v in outputs.values())

print(f"input_bits={input_bits}, state_bits={state_bits}, out_bits={out_bits}")

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

# Initialize inputs
for i in range(input_bits):
    in_arr[i] = ZEROS

# Set reset to 1
_set_field_bits_bitsliced(in_arr, lsb=int(inputs["reset"]["lsb"]), width=int(inputs["reset"]["width"]), value=1)

# Check state before
state_nonzero_before = sum(1 for i in range(state_bits) if int(st0_arr[i][0]) != 0)
print(f"State non-zero bits before reset: {state_nonzero_before}")

# Run one step
lib.circuit_steps_shared(in_ptr, out_ptr, st0_ptr, st1_ptr, 1)

# Check state after
state_nonzero_after = sum(1 for i in range(state_bits) if int(st1_arr[i][0]) != 0)
print(f"State non-zero bits after reset: {state_nonzero_after}")

# Check out_ready
out_ready_lsb = int(outputs["out_ready"]["lsb"])
out_ready_bit = 1 if (int(out_arr[out_ready_lsb][0]) & 1) else 0
print(f"out_ready: {out_ready_bit}")
