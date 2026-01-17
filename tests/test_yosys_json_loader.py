import json
import tempfile
import unittest
from pathlib import Path

from stc.yosys_json import load_design


class TestYosysJsonLoader(unittest.TestCase):
    def test_load_minimal_design(self) -> None:
        design = {
            "modules": {
                "top": {
                    "ports": {
                        "a": {"direction": "input", "bits": [2]},
                        "y": {"direction": "output", "bits": [3]},
                    },
                    "cells": {},
                }
            }
        }

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "normalized.json"
            p.write_text(json.dumps(design), encoding="utf-8")
            loaded = load_design(p)

        self.assertEqual(loaded.top, "top")
        self.assertIn("top", loaded.modules)
        self.assertIn("a", loaded.modules["top"].ports)
        self.assertEqual(loaded.modules["top"].ports["a"].bits, [2])

    def test_load_cell(self) -> None:
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
                            "parameters": {},
                        }
                    },
                }
            }
        }

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "normalized.json"
            p.write_text(json.dumps(design), encoding="utf-8")
            loaded = load_design(p)

        cell = loaded.modules["top"].cells["$not$0"]
        self.assertEqual(cell.type, "$not")
        self.assertEqual(cell.connections["A"], [2])
