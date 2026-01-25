from __future__ import annotations

import ctypes
from dataclasses import dataclass


@dataclass(frozen=True)
class CudaError(Exception):
    message: str
    code: int | None = None

    def __str__(self) -> str:
        if self.code is None:
            return self.message
        return f"{self.message} (code={self.code})"


class Cuda:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcuda.so")

        self.lib.cuInit.argtypes = [ctypes.c_uint]
        self.lib.cuInit.restype = ctypes.c_int

        self.lib.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.lib.cuDeviceGetCount.restype = ctypes.c_int

        self.lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.lib.cuDeviceGet.restype = ctypes.c_int

        self.lib.cuDeviceComputeCapability.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        self.lib.cuDeviceComputeCapability.restype = ctypes.c_int

        self.lib.cuCtxCreate_v2.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
            ctypes.c_int,
        ]
        self.lib.cuCtxCreate_v2.restype = ctypes.c_int

        self.lib.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
        self.lib.cuCtxDestroy_v2.restype = ctypes.c_int

        self.lib.cuModuleLoadDataEx.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.cuModuleLoadDataEx.restype = ctypes.c_int

        self.lib.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        self.lib.cuModuleGetFunction.restype = ctypes.c_int

        self.lib.cuMemAlloc_v2.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]
        self.lib.cuMemAlloc_v2.restype = ctypes.c_int

        self.lib.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        self.lib.cuMemFree_v2.restype = ctypes.c_int

        self.lib.cuMemcpyHtoD_v2.argtypes = [
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.lib.cuMemcpyHtoD_v2.restype = ctypes.c_int

        self.lib.cuMemcpyDtoH_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_size_t,
        ]
        self.lib.cuMemcpyDtoH_v2.restype = ctypes.c_int

        self.lib.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        self.lib.cuLaunchKernel.restype = ctypes.c_int

        self.lib.cuCtxSynchronize.argtypes = []
        self.lib.cuCtxSynchronize.restype = ctypes.c_int

        self.lib.cuEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        self.lib.cuEventCreate.restype = ctypes.c_int

        self.lib.cuEventDestroy_v2.argtypes = [ctypes.c_void_p]
        self.lib.cuEventDestroy_v2.restype = ctypes.c_int

        self.lib.cuEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.cuEventRecord.restype = ctypes.c_int

        self.lib.cuEventSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cuEventSynchronize.restype = ctypes.c_int

        self.lib.cuEventElapsedTime.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.cuEventElapsedTime.restype = ctypes.c_int

    def _check(self, code: int, msg: str) -> None:
        if code != 0:
            raise CudaError(msg, code=code)

    def init(self) -> None:
        self._check(self.lib.cuInit(0), "cuInit failed")

    def device_count(self) -> int:
        n = ctypes.c_int()
        self._check(
            self.lib.cuDeviceGetCount(ctypes.byref(n)), "cuDeviceGetCount failed"
        )
        return int(n.value)

    def device(self, idx: int) -> int:
        d = ctypes.c_int()
        self._check(
            self.lib.cuDeviceGet(ctypes.byref(d), int(idx)), "cuDeviceGet failed"
        )
        return int(d.value)

    def compute_capability(self, dev: int) -> tuple[int, int]:
        major = ctypes.c_int()
        minor = ctypes.c_int()
        self._check(
            self.lib.cuDeviceComputeCapability(
                ctypes.byref(major), ctypes.byref(minor), int(dev)
            ),
            "cuDeviceComputeCapability failed",
        )
        return (int(major.value), int(minor.value))

    def ctx_create(self, dev: int) -> ctypes.c_void_p:
        ctx = ctypes.c_void_p()
        self._check(
            self.lib.cuCtxCreate_v2(ctypes.byref(ctx), 0, int(dev)),
            "cuCtxCreate failed",
        )
        return ctx

    def ctx_destroy(self, ctx: ctypes.c_void_p) -> None:
        self._check(self.lib.cuCtxDestroy_v2(ctx), "cuCtxDestroy failed")

    def module_load_ptx(self, ptx: str) -> ctypes.c_void_p:
        mod = ctypes.c_void_p()
        buf = ctypes.create_string_buffer(ptx.encode("utf-8"))
        self._check(
            self.lib.cuModuleLoadDataEx(
                ctypes.byref(mod), ctypes.cast(buf, ctypes.c_void_p), 0, None, None
            ),
            "cuModuleLoadDataEx failed",
        )
        return mod

    def module_load_data(self, data: bytes) -> ctypes.c_void_p:
        mod = ctypes.c_void_p()
        buf = ctypes.create_string_buffer(data)
        self._check(
            self.lib.cuModuleLoadDataEx(
                ctypes.byref(mod), ctypes.cast(buf, ctypes.c_void_p), 0, None, None
            ),
            "cuModuleLoadDataEx failed",
        )
        return mod

    def module_get_function(self, mod: ctypes.c_void_p, name: str) -> ctypes.c_void_p:
        fn = ctypes.c_void_p()
        self._check(
            self.lib.cuModuleGetFunction(ctypes.byref(fn), mod, name.encode("utf-8")),
            "cuModuleGetFunction failed",
        )
        return fn

    def mem_alloc(self, nbytes: int) -> int:
        dptr = ctypes.c_uint64()
        self._check(
            self.lib.cuMemAlloc_v2(ctypes.byref(dptr), int(nbytes)), "cuMemAlloc failed"
        )
        return int(dptr.value)

    def mem_free(self, dptr: int) -> None:
        self._check(
            self.lib.cuMemFree_v2(ctypes.c_uint64(int(dptr))), "cuMemFree failed"
        )

    def memcpy_htod(self, dst: int, src: ctypes.Array, nbytes: int) -> None:
        self._check(
            self.lib.cuMemcpyHtoD_v2(
                ctypes.c_uint64(int(dst)),
                ctypes.cast(src, ctypes.c_void_p),
                int(nbytes),
            ),
            "cuMemcpyHtoD failed",
        )

    def memcpy_dtoh(self, dst: ctypes.Array, src: int, nbytes: int) -> None:
        self._check(
            self.lib.cuMemcpyDtoH_v2(
                ctypes.cast(dst, ctypes.c_void_p),
                ctypes.c_uint64(int(src)),
                int(nbytes),
            ),
            "cuMemcpyDtoH failed",
        )

    def launch(
        self,
        fn: ctypes.c_void_p,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: list[ctypes.c_void_p],
    ) -> None:
        self.launch_async(fn, grid, block, args)
        self.synchronize()

    def launch_async(
        self,
        fn: ctypes.c_void_p,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: list[ctypes.c_void_p],
    ) -> None:
        arr_t = ctypes.c_void_p * len(args)
        params = arr_t(*args)
        self._check(
            self.lib.cuLaunchKernel(
                fn,
                int(grid[0]),
                int(grid[1]),
                int(grid[2]),
                int(block[0]),
                int(block[1]),
                int(block[2]),
                0,
                None,
                params,
                None,
            ),
            "cuLaunchKernel failed",
        )

    def synchronize(self) -> None:
        self._check(self.lib.cuCtxSynchronize(), "cuCtxSynchronize failed")

    def event_create(self) -> ctypes.c_void_p:
        ev = ctypes.c_void_p()
        self._check(self.lib.cuEventCreate(ctypes.byref(ev), 0), "cuEventCreate failed")
        return ev

    def event_destroy(self, ev: ctypes.c_void_p) -> None:
        self._check(self.lib.cuEventDestroy_v2(ev), "cuEventDestroy failed")

    def event_record(self, ev: ctypes.c_void_p) -> None:
        self._check(self.lib.cuEventRecord(ev, None), "cuEventRecord failed")

    def event_synchronize(self, ev: ctypes.c_void_p) -> None:
        self._check(self.lib.cuEventSynchronize(ev), "cuEventSynchronize failed")

    def event_elapsed_ms(self, start: ctypes.c_void_p, end: ctypes.c_void_p) -> float:
        ms = ctypes.c_float()
        self._check(
            self.lib.cuEventElapsedTime(ctypes.byref(ms), start, end),
            "cuEventElapsedTime failed",
        )
        return float(ms.value)
