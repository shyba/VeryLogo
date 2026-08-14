import unittest
from pathlib import Path

from stc.sched import (
    CpuModelError,
    RegisterFileSpec,
    ResourceSpec,
    TargetModel,
    cpu_from_agner_csv,
    load_agner_forms,
    parse_pipe_set,
)


TABLE = (
    Path(__file__).resolve().parents[1]
    / "inference"
    / "results"
    / "agner_zen5_avx512.csv"
)


class TestCpuModel(unittest.TestCase):
    def test_pipe_parser_preserves_eligible_ports(self):
        self.assertEqual(parse_pipe_set("P0123"), ("P0", "P1", "P2", "P3"))
        self.assertEqual(parse_pipe_set("P01"), ("P0", "P1"))
        self.assertEqual(parse_pipe_set("P0/P1"), ("P0", "P1"))

    def test_loader_preserves_every_operand_form(self):
        forms = load_agner_forms(TABLE)
        self.assertEqual(len(forms), 27)
        self.assertEqual(len([form for form in forms if form.family == "vnni8"]), 1)
        self.assertEqual(len([form for form in forms if form.mnemonic == "VPERMD"]), 2)

    def test_form_selection_requires_operands_when_ambiguous(self):
        cpu = cpu_from_agner_csv(TABLE)
        with self.assertRaises(CpuModelError):
            cpu.select_form("VPERMD")
        z_form = cpu.select_form("VPERMD", operands="z,z,z")
        self.assertEqual(z_form.latency, 5.0)

    def test_form_keeps_unknown_memory_latency_and_chain_bound(self):
        cpu = cpu_from_agner_csv(TABLE)
        memory_form = cpu.select_form("add/sub", operands="v,m")
        self.assertIsNone(memory_form.latency)
        vnni = cpu.select_form("vnni8")
        self.assertEqual(vnni.pipes, ("P0", "P1"))
        self.assertEqual(vnni.minimum_independent_chains, 8)
        self.assertIn("avx512vnni", vnni.features)

    def test_machine_manifest_is_explicit(self):
        cpu = cpu_from_agner_csv(
            TABLE,
            name="zen5-avx512-test",
            issue_width=4,
            resources=(ResourceSpec("P0"), ResourceSpec("P1")),
            register_files=(RegisterFileSpec("zmm", 32, 512),),
            features=("avx512f",),
        )
        self.assertEqual(cpu.issue_width, 4)
        self.assertEqual(cpu.resource_names, ("P0", "P1"))
        self.assertEqual(cpu.register_file("zmm").count, 32)
        self.assertIn("avx512f", cpu.features)

        target = TargetModel(
            "zen5-floor",
            registers=32,
            issue_width=4,
            cpu=cpu,
        )
        self.assertIs(target.cpu, cpu)
        self.assertEqual(target.max_per_cycle("unknown"), 4)
