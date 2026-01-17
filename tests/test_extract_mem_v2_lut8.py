import json
import tempfile
import unittest
from pathlib import Path

from stc.extract import extract_tick_ir
from stc.interp import reset_state, tick
from stc.yosys_json import load_design


def _encode_mem_init(table: list[int]) -> str:
    bits_lsb_first: list[str] = []
    for v in table:
        for k in range(8):
            bits_lsb_first.append("1" if ((v >> k) & 1) else "0")
    return "".join(reversed(bits_lsb_first))


class TestExtractMemV2Lut8(unittest.TestCase):
    def test_extract_mem_v2_rom_as_lut8(self) -> None:
        table = list(range(256))
        init = _encode_mem_init(table)
        design = {
            "modules": {
                "top": {
                    "ports": {
                        "x": {"direction": "input", "bits": [2, 3, 4, 5, 6, 7, 8, 9]},
                        "y": {
                            "direction": "output",
                            "bits": [10, 11, 12, 13, 14, 15, 16, 17],
                        },
                    },
                    "cells": {
                        "rom": {
                            "type": "$mem_v2",
                            "port_directions": {
                                "RD_ADDR": "input",
                                "RD_EN": "input",
                                "RD_DATA": "output",
                            },
                            "connections": {
                                "RD_ADDR": [2, 3, 4, 5, 6, 7, 8, 9],
                                "RD_EN": [1],
                                "RD_DATA": [10, 11, 12, 13, 14, 15, 16, 17],
                            },
                            "parameters": {
                                "ABITS": "8",
                                "WIDTH": "8",
                                "SIZE": "256",
                                "RD_PORTS": "1",
                                "WR_PORTS": "0",
                                "RD_CLK_ENABLE": "0",
                                "INIT": init,
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

        ir = extract_tick_ir(yosys)
        state = reset_state(ir)
        for x in (0, 1, 2, 7, 42, 255):
            _, out = tick(ir, state, {"x": x})
            self.assertEqual(out["y"], x)
