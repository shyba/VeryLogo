import json
import tempfile
import unittest
from pathlib import Path

from stc.extract import extract_tick_ir
from stc.interp import reset_state, tick
from stc.tick_ir import BitVecType, Mul, Not, Slice, Var
from stc.yosys_json import load_design
from stc.yosys_frontend import run_yosys


class TestExtract(unittest.TestCase):
    def test_extract_not(self) -> None:
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
            yosys = load_design(p)

        ir = extract_tick_ir(yosys)
        self.assertEqual(ir.output_exprs["y"], Not(x=Var(name="a")))

    def test_extract_add_4bit(self) -> None:
        design = {
            "modules": {
                "top": {
                    "ports": {
                        "a": {"direction": "input", "bits": [2, 3, 4, 5]},
                        "b": {"direction": "input", "bits": [6, 7, 8, 9]},
                        "y": {"direction": "output", "bits": [10, 11, 12, 13]},
                    },
                    "cells": {
                        "$add$0": {
                            "type": "$add",
                            "port_directions": {
                                "A": "input",
                                "B": "input",
                                "Y": "output",
                            },
                            "connections": {
                                "A": [2, 3, 4, 5],
                                "B": [6, 7, 8, 9],
                                "Y": [10, 11, 12, 13],
                            },
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

        ir = extract_tick_ir(yosys)
        state = reset_state(ir)
        _, outputs = tick(ir, state, {"a": 1, "b": 2})
        self.assertEqual(outputs["y"], 3)

    def test_extract_mul_4bit(self) -> None:
        design = {
            "modules": {
                "top": {
                    "ports": {
                        "a": {"direction": "input", "bits": [2, 3, 4, 5]},
                        "b": {"direction": "input", "bits": [6, 7, 8, 9]},
                        "y": {"direction": "output", "bits": [10, 11, 12, 13]},
                    },
                    "cells": {
                        "$mul$0": {
                            "type": "$mul",
                            "port_directions": {
                                "A": "input",
                                "B": "input",
                                "Y": "output",
                            },
                            "connections": {
                                "A": [2, 3, 4, 5],
                                "B": [6, 7, 8, 9],
                                "Y": [10, 11, 12, 13],
                            },
                            "parameters": {
                                "A_WIDTH": "4",
                                "B_WIDTH": "4",
                                "Y_WIDTH": "4",
                                "A_SIGNED": "0",
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

        ir = extract_tick_ir(yosys)
        self.assertEqual(
            ir.inputs, {"a": BitVecType(width=4), "b": BitVecType(width=4)}
        )
        self.assertEqual(ir.outputs, {"y": BitVecType(width=4)})
        self.assertEqual(
            ir.output_exprs["y"],
            Slice(x=Mul(a=Var(name="a"), b=Var(name="b")), offset=0, width=4),
        )

        state = reset_state(ir)
        _, outputs = tick(ir, state, {"a": 7, "b": 9})
        self.assertEqual(outputs["y"], 15)

    def test_extract_mul_extends_operands_to_output_width(self) -> None:
        design = {
            "modules": {
                "top": {
                    "ports": {
                        "a": {"direction": "input", "bits": [2, 3]},
                        "b": {"direction": "input", "bits": [4, 5]},
                        "y": {"direction": "output", "bits": [6, 7, 8, 9]},
                    },
                    "cells": {
                        "$mul$0": {
                            "type": "$mul",
                            "port_directions": {
                                "A": "input",
                                "B": "input",
                                "Y": "output",
                            },
                            "connections": {
                                "A": [2, 3],
                                "B": [4, 5],
                                "Y": [6, 7, 8, 9],
                            },
                            "parameters": {
                                "A_WIDTH": "2",
                                "B_WIDTH": "2",
                                "Y_WIDTH": "4",
                                "A_SIGNED": "0",
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

        ir = extract_tick_ir(yosys)
        expr = ir.output_exprs["y"]
        self.assertIsInstance(expr, Slice)
        self.assertIsInstance(expr.x, Mul)
        self.assertEqual(ir.outputs["y"], BitVecType(width=4))
        state = reset_state(ir)
        _, outputs = tick(ir, state, {"a": 3, "b": 3})
        self.assertEqual(outputs["y"], 9)

    def test_extract_mul_from_yosys(self) -> None:
        source = """
module top(
    input wire [3:0] a,
    input wire [3:0] b,
    output wire [7:0] y
);
    assign y = a * b;
endmodule
"""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            input_v = root / "mul.v"
            normalized = root / "normalized.json"
            input_v.write_text(source, encoding="utf-8")
            run_yosys(input_v, normalized, top="top")
            ir = extract_tick_ir(load_design(normalized, top="top"))

        self.assertIsInstance(ir.output_exprs["y"], Slice)
        self.assertIsInstance(ir.output_exprs["y"].x, Mul)
        state = reset_state(ir)
        _, outputs = tick(ir, state, {"a": 15, "b": 15})
        self.assertEqual(outputs["y"], 225)

    def test_extract_dff_state(self) -> None:
        design = {
            "modules": {
                "top": {
                    "ports": {
                        "clk": {"direction": "input", "bits": [1]},
                        "i": {"direction": "input", "bits": [2]},
                        "o": {"direction": "output", "bits": [4]},
                    },
                    "cells": {
                        "$dff$0": {
                            "type": "$dff",
                            "port_directions": {
                                "CLK": "input",
                                "D": "input",
                                "Q": "output",
                            },
                            "connections": {"CLK": [1], "D": [2], "Q": [3]},
                            "parameters": {},
                        },
                        "$not$0": {
                            "type": "$not",
                            "port_directions": {"A": "input", "Y": "output"},
                            "connections": {"A": [3], "Y": [4]},
                            "parameters": {},
                        },
                    },
                }
            }
        }

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "normalized.json"
            p.write_text(json.dumps(design), encoding="utf-8")
            yosys = load_design(p)

        ir = extract_tick_ir(yosys)
        self.assertNotIn("clk", ir.inputs)
        state = reset_state(ir)
        state1, out0 = tick(ir, state, {"i": 1})
        self.assertEqual(out0["o"], 1)
        _, out1 = tick(ir, state1, {"i": 1})
        self.assertEqual(out1["o"], 0)

    def test_extract_sdff_reset_value(self) -> None:
        design = {
            "modules": {
                "top": {
                    "ports": {
                        "clk": {"direction": "input", "bits": [1]},
                        "rst": {"direction": "input", "bits": [2]},
                        "i": {"direction": "input", "bits": [3]},
                        "o": {"direction": "output", "bits": [5]},
                    },
                    "cells": {
                        "$sdff$0": {
                            "type": "$sdff",
                            "port_directions": {
                                "CLK": "input",
                                "SRST": "input",
                                "D": "input",
                                "Q": "output",
                            },
                            "connections": {
                                "CLK": [1],
                                "SRST": [2],
                                "D": [3],
                                "Q": [4],
                            },
                            "parameters": {
                                "CLK_POLARITY": "1",
                                "SRST_POLARITY": "1",
                                "SRST_VALUE": "1",
                                "WIDTH": "1",
                            },
                        },
                        "$not$0": {
                            "type": "$not",
                            "port_directions": {"A": "input", "Y": "output"},
                            "connections": {"A": [4], "Y": [5]},
                            "parameters": {},
                        },
                    },
                }
            }
        }

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "normalized.json"
            p.write_text(json.dumps(design), encoding="utf-8")
            yosys = load_design(p)

        ir = extract_tick_ir(yosys)
        self.assertIn("rst", ir.inputs)
        state = reset_state(ir)
        self.assertTrue(next(iter(state.state.values())))
        state1, out0 = tick(ir, state, {"rst": False, "i": 0})
        self.assertEqual(out0["o"], 0)
        _, out1 = tick(ir, state1, {"rst": False, "i": 0})
        self.assertEqual(out1["o"], 1)
