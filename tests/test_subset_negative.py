import json
import tempfile
import unittest
from pathlib import Path

from stc.subset import SubsetError, check_subset
from stc.yosys_json import load_design


class TestSubsetNegative(unittest.TestCase):
    def test_reject_unknown_cell(self) -> None:
        design = {
            "modules": {
                "top": {
                    "ports": {},
                    "cells": {
                        "$foo$0": {
                            "type": "$foo",
                            "port_directions": {},
                            "connections": {},
                            "parameters": {},
                        }
                    },
                }
            }
        }

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "normalized.json"
            p.write_text(json.dumps(design), encoding="utf-8")
            yosys = load_design(p)

        with self.assertRaises(SubsetError):
            check_subset(yosys)

    def test_reject_sdff_missing_ports(self) -> None:
        design = {
            "modules": {
                "top": {
                    "ports": {},
                    "cells": {
                        "$sdff$0": {
                            "type": "$sdff",
                            "port_directions": {"D": "input", "Q": "output"},
                            "connections": {"D": [2], "Q": [3]},
                            "parameters": {"WIDTH": "1"},
                        }
                    },
                }
            }
        }

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "normalized.json"
            p.write_text(json.dumps(design), encoding="utf-8")
            yosys = load_design(p)

        with self.assertRaises(SubsetError):
            check_subset(yosys)

    def test_reject_signed_mul(self) -> None:
        design = {
            "modules": {
                "top": {
                    "ports": {},
                    "cells": {
                        "$mul$0": {
                            "type": "$mul",
                            "port_directions": {
                                "A": "input",
                                "B": "input",
                                "Y": "output",
                            },
                            "connections": {"A": [2], "B": [3], "Y": [4]},
                            "parameters": {
                                "A_SIGNED": "1",
                                "B_SIGNED": "0",
                            },
                        }
                    },
                }
            }
        }

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "normalized.json"
            p.write_text(json.dumps(design), encoding="utf-8")
            yosys = load_design(p)

        with self.assertRaisesRegex(SubsetError, "signed cell params"):
            check_subset(yosys)
