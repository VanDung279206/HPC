@echo off
REM build_msvc.bat  –  Build với MSVC (Visual Studio 2019/2022)
REM Chạy trong "Developer Command Prompt for VS"

set SRC=src\cpu\cpu_mc.cpp
set OUT=cpu_mc.exe
set ROOT=%~dp0

echo [BUILD] Compiling %SRC% with MSVC ...
cl /O2 /openmp /std:c++17 /EHsc /W3 %SRC% /Fe:%OUT% /link
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Compile failed!
    exit /b 1
)
echo [OK] Build successful: %OUT%

echo.
echo [RUN] Running Monte Carlo simulation ...
%OUT% "%ROOT%"
