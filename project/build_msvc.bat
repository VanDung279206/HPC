@echo off
REM build_msvc.bat  -  Build voi MSVC (Visual Studio 2019/2022/2026)
REM Chay trong "Developer Command Prompt for VS"
REM
REM Usage:
REM   build_msvc.bat                 -- build + run voi JSON config
REM   build_msvc.bat --M 10000000    -- ghi de M qua CLI

set SRC=src\cpu\cpu_mc.cpp
set OUT=cpu_mc.exe
set ROOT=%~dp0
if "%ROOT:~-1%"=="\" set ROOT=%ROOT:~0,-1%

set EXTRA_ARGS=%*

echo [BUILD] Compiling %SRC% with MSVC ...
cl /O2 /openmp /std:c++17 /EHsc /W3 %SRC% /Fe:%OUT% /link
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Compile failed!
    exit /b 1
)
echo [OK] Build successful: %OUT%

echo.
echo [RUN] Running Monte Carlo simulation ...
%OUT% "%ROOT%" %EXTRA_ARGS%
