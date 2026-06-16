# Build bench_ampere.exe here (sm_89), copy to the 4050 box, run it.
# Usage:  powershell -File tests\iter.ps1 [m n k R iters]
Set-Location 'C:\Users\ADMIN\audits\p40-alpha-miner\p40-pearl-gemm'
$vc   = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
$nvcc = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe'
$flags = '-O3 -std=c++20 --expt-relaxed-constexpr --expt-extended-lambda -arch=sm_89 -cudart static -Xcompiler /MT'
Remove-Item tests\bench_ampere.exe -ErrorAction SilentlyContinue   # so a failed build can't run stale
cmd /c "`"$vc`" >nul 2>&1 && `"$nvcc`" $flags -o tests\bench_ampere.exe tests\bench_ampere.cu 2>&1"
if (-not (Test-Path tests\bench_ampere.exe)) { Write-Output 'BUILD FAILED'; exit 1 }
$key = "$env:USERPROFILE\.ssh\id_rsa"
$h   = 'kfn collegiate@4.tcp.us-cal-1.ngrok.io'
scp -i $key -P 24984 -o BatchMode=yes tests\bench_ampere.exe "${h}:C:/obm/bench_ampere.exe" 2>&1 | Out-Null
ssh -i $key -p 24984 -o BatchMode=yes $h "C:\obm\bench_ampere.exe $args"
