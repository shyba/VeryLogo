import unittest

from stc.schedule_stats_bin import read_schedule_stats_bin, write_schedule_stats_bin


class TestScheduleStatsBin(unittest.TestCase):
    def test_roundtrip(self) -> None:
        stats = {
            "total_cycles": 10,
            "max_live": 5,
            "num_spills": 1,
            "gates": 20,
            "inputs": 3,
            "outputs": 1,
            "target": "avx512",
            "scheduler": "list",
        }
        path = "out/test_schedule_stats.bin"
        write_schedule_stats_bin(stats, path)
        loaded = read_schedule_stats_bin(path)
        for key in ("total_cycles", "max_live", "num_spills", "gates", "inputs", "outputs"):
            self.assertEqual(stats[key], loaded[key])
        self.assertEqual(stats["target"], loaded["target"])
        self.assertEqual(stats["scheduler"], loaded["scheduler"])
