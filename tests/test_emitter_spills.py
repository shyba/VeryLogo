import subprocess
import tempfile
import unittest

from stc.sched import AVX512, list_schedule
from stc.sched.emit import AVX2Emitter
from stc.sched.liveness import compute_live_ranges
from stc.sched.regalloc import allocate_registers


class TestEmitterWithSpills(unittest.TestCase):
    def _gcc_supports_avx2(self) -> bool:
        with tempfile.TemporaryDirectory() as td:
            c_path = f"{td}/probe.c"
            o_path = f"{td}/probe.o"
            with open(c_path, "w") as f:
                f.write(
                    "#include <immintrin.h>\n"
                    "int main(){ __m256i x=_mm256_setzero_si256(); (void)x; return 0; }\n"
                )
            res = subprocess.run(
                ["gcc", "-mavx2", "-c", c_path, "-o", o_path],
                capture_output=True,
                text=True,
            )
            return res.returncode == 0

    def test_emitter_never_uses_negative_registers_under_spills(self) -> None:
        # Construct a circuit that forces spills by constraining the allocator to
        # very few registers while keeping many values live until the end.
        input_bits = 24
        gates = []
        for i in range(48):
            a = i % input_bits
            b = (i + 1) % input_bits
            # New node indices are implicit: input_bits + gate_index
            gates.append(("xor", a, b))

        outputs = [(i, False) for i in range(input_bits)]
        outputs.extend((input_bits + g_idx, False) for g_idx in range(len(gates)))

        schedule = list_schedule(gates, input_bits, outputs, AVX512, "slack")
        ranges = compute_live_ranges(schedule, gates, input_bits, outputs)
        allocation = allocate_registers(ranges, schedule, num_registers=4)
        self.assertGreater(allocation.num_spills, 0)

        code = AVX2Emitter().emit(schedule, allocation, gates, input_bits, outputs, "c")

        # Regression: emitter previously produced r-1 when an operand was spilled
        # and reloaded into a temp register.
        self.assertNotIn("r-", code)

        if not self._gcc_supports_avx2():
            self.skipTest("gcc does not support -mavx2 in this environment")

        with tempfile.TemporaryDirectory() as td:
            c_path = f"{td}/test.c"
            o_path = f"{td}/test.o"
            with open(c_path, "w") as f:
                f.write(code)
            res = subprocess.run(
                ["gcc", "-O2", "-mavx2", "-c", c_path, "-o", o_path],
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 0, res.stderr[:500])

