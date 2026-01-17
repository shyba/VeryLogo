import json
import tempfile
import unittest
from pathlib import Path

from stc.cli import run_pipeline
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
            normalized.write_text(json.dumps(design), encoding="utf-8")
            out_dir = base / "out"

            run_pipeline(normalized, out_dir, bound=2)

            self.assertTrue((out_dir / "tick_ir.json").exists())
            self.assertTrue((out_dir / "reduced_tick_ir.json").exists())
            self.assertTrue((out_dir / "avr.c").exists())

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
