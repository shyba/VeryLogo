import unittest

from stc.interp import eval_expr
from stc.tick_ir import (
    BitVecConst,
    BitVecType,
    SimdExtractLane,
    SimdInsertLane,
    SimdSplat,
    SimdType,
    Var,
)


class TestSimdLaneOps(unittest.TestCase):
    def test_splat(self) -> None:
        lane = BitVecType(width=4)
        vec = SimdType(lane_width=4, lanes=2)
        types = {"x": lane}
        expr = SimdSplat(to=vec, x=Var("x"))
        self.assertEqual(eval_expr(expr, types, {"x": 0xA}), 0xAA)

    def test_extract_lane(self) -> None:
        vec = SimdType(lane_width=4, lanes=2)
        types = {"v": vec}
        self.assertEqual(
            eval_expr(SimdExtractLane(x=Var("v"), lane=0), types, {"v": 0xAB}), 0xB
        )
        self.assertEqual(
            eval_expr(SimdExtractLane(x=Var("v"), lane=1), types, {"v": 0xAB}), 0xA
        )

    def test_insert_lane(self) -> None:
        vec = SimdType(lane_width=4, lanes=2)
        lane = BitVecType(width=4)
        types = {"v": vec, "x": lane}
        self.assertEqual(
            eval_expr(
                SimdInsertLane(x=Var("v"), lane=0, value=Var("x")),
                types,
                {"v": 0xAB, "x": 0x5},
            ),
            0xA5,
        )
        self.assertEqual(
            eval_expr(
                SimdInsertLane(x=Var("v"), lane=1, value=BitVecConst(width=4, value=5)),
                types,
                {"v": 0xAB, "x": 0},
            ),
            0x5B,
        )
