"""Phase 3 gate: the torch-free transpose kernel (plain + column-slice) matches
torch .t().contiguous(). Run: python tests/test_capi_phase3.py
"""
import ctypes
import os
import sys

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DLL = os.path.join(HERE, "p40cuda.dll" if sys.platform == "win32" else "libp40cuda.so")
lib = ctypes.CDLL(DLL)
lib.p40_transpose_i8.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int] * 4
lib.p40_sync.argtypes = []
dev = torch.device("cuda", 0)
ok = True


def T(src, rows, cols, src_ld, col_off, out_shape):
    dst = torch.empty(out_shape, dtype=torch.int8, device=dev)
    assert lib.p40_transpose_i8(src.data_ptr(), dst.data_ptr(), rows, cols, src_ld, col_off) == 0
    lib.p40_sync()
    return dst


# plain transpose [R,K] -> [K,R] (e.g. EAR.t())
R, K = 256, 4096
src = torch.randint(-128, 127, (R, K), dtype=torch.int8, device=dev)
d = T(src, R, K, K, 0, (K, R))
m1 = torch.equal(d, src.t().contiguous())
print(f"  [{'PASS' if m1 else 'FAIL'}] plain transpose [{R},{K}]->[{K},{R}]")
ok &= m1

# column-slice transpose: B[K,N][:, c0:c0+RS].t() -> [RS,K]  (Bt_cols)
Kk, Nn, RS, c0 = 4096, 8192, 4096, 4096
B = torch.randint(-128, 127, (Kk, Nn), dtype=torch.int8, device=dev)
d2 = T(B, Kk, RS, Nn, c0, (RS, Kk))
ref = B[:, c0:c0 + RS].t().contiguous()
m2 = torch.equal(d2, ref)
print(f"  [{'PASS' if m2 else 'FAIL'}] column-slice transpose B[:,{c0}:{c0+RS}].t() -> [{RS},{Kk}]")
ok &= m2

print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
sys.exit(0 if ok else 1)
