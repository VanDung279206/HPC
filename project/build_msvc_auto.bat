@echo off
REM build_msvc_auto.bat - Build MSVC tu dong bang vswhere
REM Ho tro moi phien ban VS (2017, 2019, 2022, 2026, Build Tools, custom path)
REM
REM Usage:
REM   build_msvc_auto.bat                    -- build + run voi JSON config
REM   build_msvc_auto.bat --M 10000000       -- ghi de M qua CLI
REM   build_msvc_auto.bat --threads 8 --M 1000000

set SRC=%~dp0src\cpu\cpu_mc.cpp
set OUT=%~dp0cpu_mc.exe
set ROOT=%~dp0
if "%ROOT:~-1%"=="\" set ROOT=%ROOT:~0,-1%

REM Collect any extra args to pass to cpu_mc
set EXTRA_ARGS=%*

REM Tim vswhere.exe
set VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe
if not exist "%VSWHERE%" set VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe

if not exist "%VSWHERE%" (
    echo [ERROR] Khong tim thay vswhere.exe
    echo Vui long cai dat Visual Studio hoac Build Tools tu:
    echo https://visualstudio.microsoft.com/downloads/
    exit /b 1
)

REM Dung vswhere de tim installation co bo cong cu C++ (x64)
set VCVARS=
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualCpp.Tools.HostX64.TargetX64 -find VC\Auxiliary\Build\vcvars64.bat`) do (
    set "VCVARS=%%i"
)

if "%VCVARS%"=="" goto :notfound
goto :found

:notfound
echo [ERROR] Khong tim thay bo cong cu C++ (Desktop development with C++).
echo Vui long mo Visual Studio Installer, chon Modify va cai dat Desktop development with C++.
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

if exist "%~dp0cpu_mc.obj" del "%~dp0cpu_mc.obj" >nul 2>&1

echo.
echo [RUN] Running Monte Carlo simulation ...
echo [RUN] Project root: %ROOT%
if defined EXTRA_ARGS (
    echo [RUN] Extra args: %EXTRA_ARGS%
)
"%OUT%" "%ROOT%" %EXTRA_ARGS%
