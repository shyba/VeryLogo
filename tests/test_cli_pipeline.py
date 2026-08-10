import json
import tempfile
import unittest
from pathlib import Path

from stc.cli import run_pipeline
from stc.tick_ir_bin2 import read_tick_ir_bin
from stc.tooling import ToolMissing


class TestCliPipeline(unittest.TestCase):
    def test_pipeline_outputs(self) -> None:
        design = {
            "modules": {
                "top": {
                    "ports": {
                        "a": {"direction": "input", "bits": [2]},
                        "y": {"direction": "output", "bits": [3]},
                    },
                    "cells": {
                        "$not$0": {
                            "type": "$not",
                            "port_directions": {"A": "input", "Y": "output"},
                            "connections": {"A": [2], "Y": [3]},
                        }
                    },
                }
            }
        }

        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            normalized = base / "normalized.json"
            normalized.write_text(__import__("json").dumps(design), encoding="utf-8")
            out_dir = base / "out"

            run_pipeline(normalized, out_dir, bound=2)

            self.assertTrue((out_dir / "tick_ir.bin").exists())
            self.assertTrue((out_dir / "reduced_tick_ir.bin").exists())
            self.assertTrue((out_dir / "avr.c").exists())

    def test_pipeline_fuse_ticks_replicate_inputs(self) -> None:
        design = {
            "modules": {
                "top": {
                    "ports": {
                        "a": {"direction": "input", "bits": [2]},
                        "y": {"direction": "output", "bits": [3]},
                    },
                    "cells": {
                        "$not$0": {
                            "type": "$not",
                            "port_directions": {"A": "input", "Y": "output"},
                            "connections": {"A": [2], "Y": [3]},
                        }
                    },
                }
            }
        }

        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            normalized = base / "normalized.json"
            normalized.write_text(__import__("json").dumps(design), encoding="utf-8")
            out_dir = base / "out"

            run_pipeline(
                normalized,
                out_dir,
                bound=2,
                fuse_ticks=3,
                fuse_mode="final",
                fuse_input_policy="replicate",
            )

            reduced = read_tick_ir_bin(out_dir / "reduced_tick_ir.bin")
            self.assertIn("a__t2", reduced.inputs)
            self.assertIn("y", reduced.output_exprs)
            y = reduced.output_exprs["y"]
            from stc.tick_ir import Not, Var

            self.assertIsInstance(y, Not)
            self.assertIsInstance(y.x, Var)
            self.assertEqual(y.x.name, "a__t2")

    def test_pipeline_ptx_backend_emits_ptx(self) -> None:
        design = {
            "modules": {
                "top": {
                    "ports": {
                        "a": {"direction": "input", "bits": [2]},
                        "y": {"direction": "output", "bits": [3]},
                    },
                    "cells": {
                        "$not$0": {
                            "type": "$not",
                            "port_directions": {"A": "input", "Y": "output"},
                            "connections": {"A": [2], "Y": [3]},
                        }
                    },
                }
            }
        }

        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            normalized = base / "normalized.json"
            normalized.write_text(__import__("json").dumps(design), encoding="utf-8")
            out_dir = base / "out"

            run_pipeline(normalized, out_dir, bound=2, backend="ptx")

            out_path = out_dir / "circuit_ptx.ptx"
            self.assertTrue(out_path.exists())
            self.assertFalse((out_dir / "avr.c").exists())
            code = out_path.read_text(encoding="utf-8")
            self.assertIn(".visible .entry", code)

    def test_pipeline_futhark_backend_emits_futhark(self) -> None:
        design = {
            "modules": {
                "top": {
                    "ports": {
                        "a": {"direction": "input", "bits": [2]},
                        "y": {"direction": "output", "bits": [3]},
                    },
                    "cells": {
                        "$not$0": {
                            "type": "$not",
                            "port_directions": {"A": "input", "Y": "output"},
                            "connections": {"A": [2], "Y": [3]},
                        }
                    },
                }
            }
        }

        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            normalized = base / "normalized.json"
            normalized.write_text(__import__("json").dumps(design), encoding="utf-8")
            out_dir = base / "out"

            run_pipeline(normalized, out_dir, bound=2, backend="futhark")

            fut_path = out_dir / "circuit_futhark.fut"
            manifest_path = out_dir / "futhark_io_manifest.json"
            metrics_path = out_dir / "futhark_source_metrics.json"
            self.assertTrue(fut_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertTrue(metrics_path.exists())
            self.assertFalse((out_dir / "avr.c").exists())
            fut_code = fut_path.read_text(encoding="utf-8")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["mode"], "combinational_fast")
            self.assertEqual(manifest["module_name"], "top")
            self.assertIn("eval_batch", manifest["entries"])
            self.assertIn("eval_batch_xor", manifest["entries"])
            self.assertIn("step_batch", manifest["entries"])
            self.assertIn("entry step_batch", fut_code)
            self.assertIn("entry eval_batch", fut_code)
            self.assertNotIn("entry run_steps_batch", fut_code)
            self.assertIsNone(manifest["fallback_reason"])
            self.assertGreaterEqual(metrics["lines"], 1)

            import shutil
            import subprocess

            if shutil.which("futhark"):
                subprocess.run(
                    ["futhark", "check", str(fut_path)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

    def test_pipeline_requires_yosys_for_verilog(self) -> None:
        import shutil

        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            input_v = base / "input.v"
            input_v.write_text("module top; endmodule\n", encoding="utf-8")
            out_dir = base / "out"
            if shutil.which("yosys"):
                run_pipeline(input_v, out_dir, top="top", bound=2)
                self.assertTrue((out_dir / "avr.c").exists())
            else:
                with self.assertRaises(ToolMissing):
                    run_pipeline(input_v, out_dir)

    def test_pipeline_region_diagnostics(self) -> None:
        design = {
            "modules": {
                "top": {
                    "ports": {
                        "a": {"direction": "input", "bits": [2, 3]},
                        "clk": {"direction": "input", "bits": [4]},
                        "rst": {"direction": "input", "bits": [5]},
                        "y": {"direction": "output", "bits": [6, 7]},
                    },
                    "cells": {
                        "$dff$0": {
                            "type": "$dff",
                            "port_directions": {
                                "CLK": "input",
                                "D": "input",
                                "Q": "output",
                            },
                            "connections": {"CLK": [4], "D": [2], "Q": [6]},
                        },
                        "$dff$1": {
                            "type": "$dff",
                            "port_directions": {
                                "CLK": "input",
                                "D": "input",
                                "Q": "output",
                            },
                            "connections": {"CLK": [4], "D": [3], "Q": [7]},
                        },
                    },
                }
            }
        }

        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            normalized = base / "normalized.json"
            normalized.write_text(json.dumps(design), encoding="utf-8")
            out_dir = base / "out"

            run_pipeline(
                normalized,
                out_dir,
                bound=2,
                backend="x86-avx512",
                use_regions=True,
                region_max_gates=5,
                region_max_boundary=10,
            )

            self.assertTrue((out_dir / "circuit_avx512_u64_regions.c").exists())
            self.assertTrue((out_dir / "regions.bin").exists())
            self.assertTrue((out_dir / "regions_stats.bin").exists())
            self.assertTrue((out_dir / "regions.dot").exists())

            from stc.region_diagnostics_bin import (
                read_regions_bin,
                read_regions_stats_bin,
            )

            regions_data = read_regions_bin(out_dir / "regions.bin")
            self.assertIn("regions", regions_data)
            self.assertIn("total_regions", regions_data)
            self.assertEqual(regions_data["ordering"], "topological")

            stats_data = read_regions_stats_bin(out_dir / "regions_stats.bin")
            self.assertIn("total_regions", stats_data)
            self.assertIn("total_gates", stats_data)
            self.assertIn("gates_per_region", stats_data)
            self.assertIn("boundary_distribution", stats_data)
            self.assertIn("critical_path_estimate", stats_data)
