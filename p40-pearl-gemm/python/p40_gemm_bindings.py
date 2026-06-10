import torch

try:
    import p40_pearl_gemm_cuda as _C
except ImportError:
    _C = None


def _check_import():
    if _C is None:
        raise ImportError(
            "p40_pearl_gemm_cuda extension not built. "
            "Run `pip install -e .` in p40-pearl-gemm/"
        )


def dp4a_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor,
    B_scales: torch.Tensor,
    C: torch.Tensor,
):
    _check_import()
    M, K = A.shape
    N, _ = B.shape
    _C.dp4a_gemm(A, B, A_scales, B_scales, C, M, N, K)


def quantize(
    input: torch.Tensor,
    output: torch.Tensor,
    scales: torch.Tensor,
    max_val: int = 63,
    smooth_scale: torch.Tensor | None = None,
    fast_math: bool = False,
):
    _check_import()
    _C.quantize(input, output, scales, max_val, smooth_scale, fast_math)


def noise_A(
    A: torch.Tensor,
    EAL: torch.Tensor,
    EAR: torch.Tensor,
    EBL: torch.Tensor,
    ApEA: torch.Tensor,
    AxEBL: torch.Tensor,
):
    _check_import()
    M, K = A.shape
    R = EAL.shape[1]
    _C.noise_A(A, EAL, EAR, EBL, ApEA, AxEBL, M, K, R)


def noise_B(
    B: torch.Tensor,
    EBR: torch.Tensor,
    EAR: torch.Tensor,
    EBL: torch.Tensor,
    BpEB: torch.Tensor,
    EARxBpEB: torch.Tensor,
):
    _check_import()
    N, K = B.shape
    R = EBR.shape[1]
    _C.noise_B(B, EBR, EAR, EBL, BpEB, EARxBpEB, N, K, R)


def denoise_converter(
    EARxBpEB_in: torch.Tensor | None = None,
    AxEBL_in: torch.Tensor | None = None,
    EARxBpEB_out: torch.Tensor | None = None,
    AxEBL_out: torch.Tensor | None = None,
):
    _check_import()
    M = AxEBL_in.shape[0] if AxEBL_in is not None else 0
    N = EARxBpEB_in.shape[0] if EARxBpEB_in is not None else 0
    R = 0
    if AxEBL_in is not None:
        R = AxEBL_in.shape[1]
    elif EARxBpEB_in is not None:
        R = EARxBpEB_in.shape[1]
    _C.denoise_converter(
        EARxBpEB_in,
        AxEBL_in,
        EARxBpEB_out,
        AxEBL_out,
        M,
        N,
        R,
    )


def inner_hash(
    input_buffer: torch.Tensor,
    iterations: int = 1,
) -> torch.Tensor:
    _check_import()
    return _C.inner_hash(input_buffer, iterations)


def noise_gen(
    R: int,
    num_threads: int = 64,
    EAL: torch.Tensor | None = None,
    EAL_fp16: torch.Tensor | None = None,
    EAR_R_major: torch.Tensor | None = None,
    EAR_K_major: torch.Tensor | None = None,
    EBL_R_major: torch.Tensor | None = None,
    EBL_K_major: torch.Tensor | None = None,
    EBR: torch.Tensor | None = None,
    EBR_fp16: torch.Tensor | None = None,
    key_A: torch.Tensor | None = None,
    key_B: torch.Tensor | None = None,
    aux_buffer: torch.Tensor | None = None,
):
    _check_import()
    _C.noise_gen(
        R,
        num_threads,
        EAL,
        EAL_fp16,
        EAR_R_major,
        EAR_K_major,
        EBL_R_major,
        EBL_K_major,
        EBR,
        EBR_fp16,
        key_A,
        key_B,
        aux_buffer,
    )


def tensor_hash(
    data: torch.Tensor,
    key: torch.Tensor,
    out: torch.Tensor,
    roots: torch.Tensor,
    threads_per_block: int = 128,
    num_stages: int = 2,
    leaves_per_mt_block: int = 512,
):
    _check_import()
    _C.tensor_hash(
        data, key, out, roots,
        threads_per_block, num_stages, leaves_per_mt_block,
    )
