@echo off
REM Run from "x64 Native Tools Command Prompt for VS 2022". Usage: build_windows.bat [sm_86]
set ARCH=%1
if "%ARCH%"=="" set ARCH=sm_86
nvcc -O3 -arch=%ARCH% -o ccl_cuda.exe src\ccl_cuda.cu
cl /O2 /Fe:ccl_cpu.exe src\ccl_cpu.c
echo Done. Try: ccl_cuda.exe --gen 1024 1024 --dump img.bin --verbose
