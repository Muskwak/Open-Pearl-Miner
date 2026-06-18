@echo off
REM Build the torch-free CUDA shared library (p40cuda.dll) on Windows.
REM Links only the CUDA runtime; driven from Python via ctypes.
REM Run from p40-pearl-gemm\ :  packaging\build_capi.bat
setlocal
cd /d "%~dp0\.."

REM Bring MSVC (cl.exe + INCLUDE/LIB) into the environment for nvcc's host step.
set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
    echo ERROR: vcvars64.bat not found at "%VCVARS%". Edit VCVARS in this script.
    exit /b 1
)
call "%VCVARS%" >nul

if not defined CUTLASS_DIR if exist "%CD%\.deps\cutlass\include" set "CUTLASS_DIR=%CD%\.deps\cutlass\include"
if not defined CUTLASS_DIR if exist "C:\Users\ADMIN\audits\aphrodite-engine\.deps\cutlass-src\include" set "CUTLASS_DIR=C:\Users\ADMIN\audits\aphrodite-engine\.deps\cutlass-src\include"
if not defined CUTLASS_DIR (
    echo ERROR: CUTLASS_DIR not found. Set CUTLASS_DIR env var or install CUTLASS.
    exit /b 1
)
set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe"

"%NVCC%" -shared -o p40cuda.dll ^
  csrc\capi\p40_capi.cu csrc\gemm\pearl_gemm_only_sm61.cu csrc\gemm\pearl_blake3_sm61.cu ^
  csrc\gemm\noising_sm61.cu csrc\gemm\noise_generation.cu csrc\blake3\blake3.cu ^
  csrc\gemm\rng_fill_sm61.cu csrc\tensor_hash\tensor_hash.cu csrc\gemm\noise_gemm_sm61.cu ^
  csrc\gemm\pearl_ampere_tc.cu ^
  -I csrc -I csrc\gemm -I csrc\blake3 -I csrc\tensor_hash -I "%CUTLASS_DIR%" ^
  -std=c++20 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math ^
  -gencode arch=compute_61,code=sm_61 ^
  -gencode arch=compute_86,code=sm_86 ^
  -gencode arch=compute_89,code=sm_89 ^
  -O3 -DNDEBUG -DP40_NO_TORCH -Xcompiler /O2
if %ERRORLEVEL% NEQ 0 exit /b 1
echo built p40cuda.dll
