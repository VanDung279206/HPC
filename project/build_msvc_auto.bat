@echo off
REM build_msvc_auto.bat - Build MSVC tu PowerShell/CMD thuong (tu tim vcvars64)
REM KHONG can mo "Developer Command Prompt" thu cong.

set SRC=%~dp0src\cpu\cpu_mc.cpp
set OUT=%~dp0cpu_mc.exe
set ROOT=%~dp0
REM Xoa dau \ cuoi (tranh double-slash khi truyen vao exe)
if "%ROOT:~-1%"=="\" set ROOT=%ROOT:~0,-1%

REM Tim vcvars64.bat tu cac phien ban VS pho bien
set VCVARS=
for %%V in (18 17 16 15) do (
    if exist "C:\Program Files\Microsoft Visual Studio\%%V\Community\VC\Auxiliary\Build\vcvars64.bat" (
        set VCVARS=C:\Program Files\Microsoft Visual Studio\%%V\Community\VC\Auxiliary\Build\vcvars64.bat
        goto :found
    )
    if exist "C:\Program Files\Microsoft Visual Studio\%%V\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
        set VCVARS=C:\Program Files\Microsoft Visual Studio\%%V\Enterprise\VC\Auxiliary\Build\vcvars64.bat
        goto :found
    )
    if exist "C:\Program Files\Microsoft Visual Studio\%%V\Professional\VC\Auxiliary\Build\vcvars64.bat" (
        set VCVARS=C:\Program Files\Microsoft Visual Studio\%%V\Professional\VC\Auxiliary\Build\vcvars64.bat
        goto :found
    )
)

:notfound
echo [ERROR] Khong tim thay Visual Studio. Cai VS 2019/2022/2026 voi C++ workload.
exit /b 1

:found
echo [INFO] Su dung: %VCVARS%
echo [BUILD] Compiling %SRC% with MSVC ...

call "%VCVARS%" >nul 2>&1
cl /O2 /openmp /std:c++17 /EHsc /W3 "%SRC%" /Fe:"%OUT%" /link
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Compile failed!
    exit /b 1
)
echo [OK] Build successful: %OUT%

echo.
echo [RUN] Running Monte Carlo simulation ...
"%OUT%" "%ROOT%"
