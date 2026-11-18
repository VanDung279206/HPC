@echo off
REM build_mingw.bat - Build voi MinGW-w64 (MSYS2)
REM
REM TRANG THAI TREN MAY NAY:
REM   Dev-Cpp g++ (C:\Program Files (x86)\Dev-Cpp\MinGW64\bin\g++.exe)
REM   => BI LOI "Illegal System DLL Relocation" - KHONG DUNG DUOC.
REM
REM De dung MinGW, cai MSYS2 tu: https://www.msys2.org/
REM Sau do chay trong MSYS2 terminal:
REM   pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-openmp
REM Roi sua GPP_PATH ben duoi thanh C:\msys64\mingw64\bin\g++.exe
REM
REM Usage:
REM   build_mingw.bat                -- build + run voi JSON config
REM   build_mingw.bat --M 10000000   -- ghi de M qua CLI

set GPP_PATH=C:\msys64\mingw64\bin\g++.exe
REM set GPP_PATH=C:\Program Files (x86)\Dev-Cpp\MinGW64\bin\g++.exe

set SRC=src\cpu\cpu_mc.cpp
set OUT=cpu_mc.exe
set ROOT=%~dp0
if "%ROOT:~-1%"=="\" set ROOT=%ROOT:~0,-1%

set EXTRA_ARGS=%*

if not exist "%GPP_PATH%" (
    echo [ERROR] Khong tim thay g++ tai: %GPP_PATH%
    echo [INFO]  Cai MSYS2 va chay: pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-openmp
    echo [INFO]  Hoac dung build_msvc_auto.bat neu da co Visual Studio.
    exit /b 1
)

echo [BUILD] Compiling %SRC% with MinGW g++ ...
"%GPP_PATH%" -std=c++17 -O2 -march=native -fopenmp -Wall %SRC% -o %OUT%
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Compile failed!
    exit /b 1
)
echo [OK] Build successful: %OUT%

echo.
echo [RUN] Running Monte Carlo simulation ...
%OUT% "%ROOT%" %EXTRA_ARGS%
