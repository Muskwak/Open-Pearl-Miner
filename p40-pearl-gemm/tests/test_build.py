"""Basic import and smoke test for p40-pearl-gemm."""
import sys
import torch


def test_import():
    from p40_pearl_gemm import dp4a_gemm, quantize, inner_hash
    print("p40_pearl_gemm imported successfully")


def test_dp4a_gemm_smoke():
    from p40_pearl_gemm import dp4a_gemm

    if not torch.cuda.is_available():
        print("CUDA not available - skipping GPU test")
        return

    device = torch.device("cuda")

    M, N, K = 256, 256, 256
    A = torch.randint(-127, 127, (M, K), dtype=torch.int8, device=device)
    B = torch.randint(-127, 127, (N, K), dtype=torch.int8, device=device)
    A_scales = torch.rand(M, dtype=torch.float32, device=device) * 0.01
    B_scales = torch.rand(N, dtype=torch.float32, device=device) * 0.01
    C = torch.empty(M, N, dtype=torch.float16, device=device)

    dp4a_gemm(A, B, A_scales, B_scales, C)
    torch.cuda.synchronize()

    assert C.shape == (M, N)
    assert C.dtype == torch.float16
    assert not torch.isnan(C).any()
    print(f"DP4A GEMM smoke test passed: C shape={C.shape}, "
          f"min={C.min().item():.4f}, max={C.max().item():.4f}")


def test_inner_hash():
    from p40_pearl_gemm import inner_hash

    if not torch.cuda.is_available():
        print("CUDA not available - skipping GPU test")
        return

    device = torch.device("cuda")
    data = torch.randint(0, 2**32 - 1, (64,), dtype=torch.uint32, device=device)
    result = inner_hash(data, iterations=1)
    torch.cuda.synchronize()
    print(f"Inner hash test passed: result={result.cpu().item():#010x}")


if __name__ == "__main__":
    test_import()
    test_inner_hash()
    test_dp4a_gemm_smoke()
    print("\nAll tests passed!")
