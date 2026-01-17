import ctypes
import shutil
import unittest

from stc.backend_ptx import emit_ptx_steps
from stc.cuda_driver import Cuda, CudaError
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
class TestBackendPtxStepsRunSm61Optional(unittest.TestCase):
    def test_steps_k_matches_k_single_steps(self) -> None:
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

        k = 7
        ptx1 = emit_ptx_steps(ir, sm="sm_61", steps=1)
        ptxk = emit_ptx_steps(ir, sm="sm_61", steps=k)

        cuda = Cuda()
        cuda.init()
        dev = cuda.device(0)
        ctx = cuda.ctx_create(dev)
        try:
            mod1 = cuda.module_load_ptx(ptx1)
            fn1 = cuda.module_get_function(mod1, "stc_eval")
            modk = cuda.module_load_ptx(ptxk)
            fnk = cuda.module_get_function(modk, "stc_eval")

            n = 256
            h_in = (ctypes.c_uint32 * (n * 1))()
            h_out = (ctypes.c_uint32 * (n * 1))()
            h_st0 = (ctypes.c_uint32 * (n * 1))()
            h_st1 = (ctypes.c_uint32 * (n * 1))()
            for i in range(n):
                h_in[i] = ctypes.c_uint32(i & 0xFF)
                h_st0[i] = ctypes.c_uint32((i * 7) & 0xFF)

            d_in = cuda.mem_alloc(ctypes.sizeof(h_in))
            d_out = cuda.mem_alloc(ctypes.sizeof(h_out))
            d_st0 = cuda.mem_alloc(ctypes.sizeof(h_st0))
            d_st1 = cuda.mem_alloc(ctypes.sizeof(h_st1))
            try:
                cuda.memcpy_htod(d_in, h_in, ctypes.sizeof(h_in))
                cuda.memcpy_htod(d_out, h_out, ctypes.sizeof(h_out))
                cuda.memcpy_htod(d_st0, h_st0, ctypes.sizeof(h_st0))
                cuda.memcpy_htod(d_st1, h_st1, ctypes.sizeof(h_st1))

                block = (128, 1, 1)
                grid = ((n + block[0] - 1) // block[0], 1, 1)

                def launch(fn: ctypes.c_void_p, st_in: int, st_out: int) -> None:
                    arg_in = ctypes.c_uint64(d_in)
                    arg_st = ctypes.c_uint64(st_in)
                    arg_out = ctypes.c_uint64(d_out)
                    arg_st2 = ctypes.c_uint64(st_out)
                    arg_n = ctypes.c_uint32(n)
                    args = [
                        ctypes.cast(ctypes.byref(arg_in), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_st), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_out), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_st2), ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(arg_n), ctypes.c_void_p),
                    ]
                    cuda.launch_async(fn, grid, block, args)

                st_in = d_st0
                st_out = d_st1
                for _ in range(k):
                    launch(fn1, st_in, st_out)
                    st_in, st_out = st_out, st_in
                cuda.synchronize()

                h_out_1 = (ctypes.c_uint32 * n)()
                h_st_1 = (ctypes.c_uint32 * n)()
                cuda.memcpy_dtoh(h_out_1, d_out, ctypes.sizeof(h_out_1))
                cuda.memcpy_dtoh(h_st_1, st_in, ctypes.sizeof(h_st_1))

                cuda.memcpy_htod(d_out, h_out, ctypes.sizeof(h_out))
                cuda.memcpy_htod(d_st0, h_st0, ctypes.sizeof(h_st0))
                cuda.memcpy_htod(d_st1, h_st1, ctypes.sizeof(h_st1))

                launch(fnk, d_st0, d_st1)
                cuda.synchronize()

                h_out_k = (ctypes.c_uint32 * n)()
                h_st_k = (ctypes.c_uint32 * n)()
                cuda.memcpy_dtoh(h_out_k, d_out, ctypes.sizeof(h_out_k))
                cuda.memcpy_dtoh(h_st_k, d_st1, ctypes.sizeof(h_st_k))

                for i in range(n):
                    self.assertEqual(int(h_out_k[i]) & 0xFF, int(h_out_1[i]) & 0xFF)
                    self.assertEqual(int(h_st_k[i]) & 0xFF, int(h_st_1[i]) & 0xFF)
            finally:
                cuda.mem_free(d_in)
                cuda.mem_free(d_out)
                cuda.mem_free(d_st0)
                cuda.mem_free(d_st1)
        except CudaError as e:
            self.skipTest(str(e))
        finally:
            cuda.ctx_destroy(ctx)
