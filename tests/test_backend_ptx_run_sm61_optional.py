import ctypes
import shutil
import unittest

from stc.backend_ptx import emit_ptx
from stc.cuda_driver import Cuda, CudaError
from stc.interp import eval_expr
from stc.tick_ir import Add, BitVecConst, BitVecType, TickIR, Var


def _can_run_sm61() -> bool:
    if not shutil.which("nvcc"):
        return False
    try:
        cuda = Cuda()
        cuda.init()
        if cuda.device_count() < 1:
            return False
        dev = cuda.device(0)
        major, minor = cuda.compute_capability(dev)
        return (major, minor) == (6, 1)
    except Exception:
        return False


@unittest.skipUnless(_can_run_sm61(), "requires CUDA driver and sm_61 device")
class TestBackendPtxRunSm61Optional(unittest.TestCase):
    def test_add_u32_matches_interpreter(self) -> None:
        t = BitVecType(width=32)
        ir = TickIR(
            name="t",
            inputs={"a": t, "b": t},
            outputs={"o": t},
            state={},
            reset_state={},
            next_state={},
            output_exprs={"o": Add(a=Var("a"), b=Var("b"))},
        )
        ptx = emit_ptx(ir, sm="sm_61")
        cuda = Cuda()
        cuda.init()
        dev = cuda.device(0)
        ctx = cuda.ctx_create(dev)
        try:
            mod = cuda.module_load_ptx(ptx)
            fn = cuda.module_get_function(mod, "stc_eval")

            n = 256
            in_words = n * 2
            out_words = n * 1
            h_in = (ctypes.c_uint32 * in_words)()
            h_out = (ctypes.c_uint32 * out_words)()
            for i in range(n):
                h_in[i * 2 + 0] = ctypes.c_uint32((i * 3) & 0xFFFFFFFF)
                h_in[i * 2 + 1] = ctypes.c_uint32((i * 7) & 0xFFFFFFFF)

            d_in = cuda.mem_alloc(ctypes.sizeof(h_in))
            d_out = cuda.mem_alloc(ctypes.sizeof(h_out))
            try:
                cuda.memcpy_htod(d_in, h_in, ctypes.sizeof(h_in))
                cuda.memcpy_htod(d_out, h_out, ctypes.sizeof(h_out))

                arg_in = ctypes.c_uint64(d_in)
                arg_out = ctypes.c_uint64(d_out)
                arg_n = ctypes.c_uint32(n)
                args = [
                    ctypes.cast(ctypes.byref(arg_in), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_out), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_n), ctypes.c_void_p),
                ]

                block = (128, 1, 1)
                grid = ((n + block[0] - 1) // block[0], 1, 1)
                cuda.launch(fn, grid, block, args)
                cuda.memcpy_dtoh(h_out, d_out, ctypes.sizeof(h_out))

                types = {"a": t, "b": t}
                for i in range(n):
                    env = {
                        "a": int(h_in[i * 2 + 0]),
                        "b": int(h_in[i * 2 + 1]),
                    }
                    exp = int(eval_expr(ir.output_exprs["o"], types, env)) & 0xFFFFFFFF
                    self.assertEqual(int(h_out[i]), exp)
            finally:
                cuda.mem_free(d_in)
                cuda.mem_free(d_out)
        except CudaError as e:
            self.skipTest(str(e))
        finally:
            cuda.ctx_destroy(ctx)

    def test_stateful_tick_matches_interpreter(self) -> None:
        t = BitVecType(width=8)
        ir = TickIR(
            name="tstate",
            inputs={"a": t},
            outputs={"o": t},
            state={"q": t},
            reset_state={"q": BitVecConst(width=8, value=0)},
            next_state={"q": Add(a=Var("q"), b=Var("a"))},
            output_exprs={"o": Var("q")},
        )
        ptx = emit_ptx(ir, sm="sm_61")
        cuda = Cuda()
        cuda.init()
        dev = cuda.device(0)
        ctx = cuda.ctx_create(dev)
        try:
            mod = cuda.module_load_ptx(ptx)
            fn = cuda.module_get_function(mod, "stc_eval")

            n = 256
            in_words = n * 1
            st_words = n * 1
            out_words = n * 1
            st2_words = n * 1
            h_in = (ctypes.c_uint32 * in_words)()
            h_st = (ctypes.c_uint32 * st_words)()
            h_out = (ctypes.c_uint32 * out_words)()
            h_st2 = (ctypes.c_uint32 * st2_words)()
            for i in range(n):
                h_in[i] = ctypes.c_uint32(i & 0xFF)
                h_st[i] = ctypes.c_uint32((i * 7) & 0xFF)

            d_in = cuda.mem_alloc(ctypes.sizeof(h_in))
            d_st = cuda.mem_alloc(ctypes.sizeof(h_st))
            d_out = cuda.mem_alloc(ctypes.sizeof(h_out))
            d_st2 = cuda.mem_alloc(ctypes.sizeof(h_st2))
            try:
                cuda.memcpy_htod(d_in, h_in, ctypes.sizeof(h_in))
                cuda.memcpy_htod(d_st, h_st, ctypes.sizeof(h_st))
                cuda.memcpy_htod(d_out, h_out, ctypes.sizeof(h_out))
                cuda.memcpy_htod(d_st2, h_st2, ctypes.sizeof(h_st2))

                arg_in = ctypes.c_uint64(d_in)
                arg_st = ctypes.c_uint64(d_st)
                arg_out = ctypes.c_uint64(d_out)
                arg_st2 = ctypes.c_uint64(d_st2)
                arg_n = ctypes.c_uint32(n)
                args = [
                    ctypes.cast(ctypes.byref(arg_in), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_st), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_out), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_st2), ctypes.c_void_p),
                    ctypes.cast(ctypes.byref(arg_n), ctypes.c_void_p),
                ]

                block = (128, 1, 1)
                grid = ((n + block[0] - 1) // block[0], 1, 1)
                cuda.launch(fn, grid, block, args)
                cuda.memcpy_dtoh(h_out, d_out, ctypes.sizeof(h_out))
                cuda.memcpy_dtoh(h_st2, d_st2, ctypes.sizeof(h_st2))

                types = {"a": t, "q": t}
                for i in range(n):
                    env = {"a": int(h_in[i]), "q": int(h_st[i])}
                    exp_o = int(eval_expr(ir.output_exprs["o"], types, env)) & 0xFF
                    exp_q = int(eval_expr(ir.next_state["q"], types, env)) & 0xFF
                    self.assertEqual(int(h_out[i]) & 0xFF, exp_o)
                    self.assertEqual(int(h_st2[i]) & 0xFF, exp_q)
            finally:
                cuda.mem_free(d_in)
                cuda.mem_free(d_st)
                cuda.mem_free(d_out)
                cuda.mem_free(d_st2)
        except CudaError as e:
            self.skipTest(str(e))
        finally:
            cuda.ctx_destroy(ctx)
