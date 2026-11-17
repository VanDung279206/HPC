# Build from any working directory; runtime defaults come from run_config.json.
[CmdletBinding()]
param(
    [ValidateSet('all','verify','mc')][string]$Mode='all',
    [Nullable[long]]$Samples=$null,
    [Nullable[int]]$BlockSize=$null,
    [Nullable[ulong]]$Seed=$null,
    [Nullable[int]]$Repeats=$null,
    [Nullable[int]]$Warmup=$null,
    [string]$OutputDirectory='',
    [string]$BuildDirectory='',
    [ValidatePattern('^[0-9]+$')][string]$Architecture='75',
    [switch]$FastMath,
    [switch]$RunTests,
    [switch]$BuildOnly
)
$ErrorActionPreference='Stop'
$projectRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (-not $BuildDirectory) { $BuildDirectory=Join-Path $projectRoot 'out/build/cuda-release' }
$buildRoot=[IO.Path]::GetFullPath($BuildDirectory)
$vsRoot=''
$programFilesX86=[Environment]::GetEnvironmentVariable('ProgramFiles(x86)')
$vswhere=Join-Path $programFilesX86 'Microsoft Visual Studio/Installer/vswhere.exe'
if (Test-Path -LiteralPath $vswhere) {
    $vsRoot=(& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1)
}
function Find-Tool([string]$Name,[string]$RelativePath) {
    $command=Get-Command $Name -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    if ($vsRoot) {
        $candidate=Join-Path $vsRoot $RelativePath
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    throw "Cannot find $Name. Install CMake/Ninja or Visual Studio's C++ CMake tools."
}
$cmake=Find-Tool 'cmake' 'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
$ninja=Find-Tool 'ninja' 'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe'
$ctest=Join-Path (Split-Path -Parent $cmake) 'ctest.exe'
$vcvars=''
if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    if (-not $vsRoot) { throw 'Open a Developer PowerShell or install Visual Studio C++ Build Tools.' }
    $vcvars=Join-Path $vsRoot 'VC/Auxiliary/Build/vcvars64.bat'
    if (-not (Test-Path -LiteralPath $vcvars)) { throw "Cannot find $vcvars" }
}
New-Item -ItemType Directory -Path $buildRoot,(Join-Path $buildRoot 'tmp') -Force | Out-Null
function Quote-Cmd([string]$Value) { return '"'+$Value.Replace('%','%%')+'"' }
$fast=if($FastMath){'ON'}else{'OFF'}
$configure=@('-S',$projectRoot,'-B',$buildRoot,'-G','Ninja',
    '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_TRY_COMPILE_CONFIGURATION=Release',
    "-DCMAKE_MAKE_PROGRAM=$ninja","-DCMAKE_CUDA_ARCHITECTURES=$Architecture",
    "-DPORTFOLIO_FAST_MATH=$fast",'-DBUILD_TESTING=ON')
$lines=@('@echo off','chcp 65001 >nul')
if ($vcvars) { $lines+=('call '+(Quote-Cmd $vcvars));$lines+='if errorlevel 1 exit /b 1' }
$lines+=((Quote-Cmd $cmake)+' '+(($configure | ForEach-Object {Quote-Cmd $_}) -join ' '))
$lines+='if errorlevel 1 exit /b 1'
$lines+=((Quote-Cmd $cmake)+' --build '+(Quote-Cmd $buildRoot)+' --config Release')
$lines+='if errorlevel 1 exit /b 1'
$batch=Join-Path $buildRoot 'build_portfolio.bat'
[IO.File]::WriteAllLines($batch,$lines,[Text.UTF8Encoding]::new($false))
$previousTemp=$env:TEMP;$previousTmp=$env:TMP
try {
    # Keep nvcc temporary files out of paths containing shell metacharacters.
    $env:TEMP=Join-Path $buildRoot 'tmp';$env:TMP=$env:TEMP
    & $env:ComSpec /d /c ('"'+$batch+'"')
    if ($LASTEXITCODE -ne 0) { throw "Build failed with exit code $LASTEXITCODE" }
    if ($RunTests) {
        & $ctest --test-dir $buildRoot -C Release --output-on-failure
        if ($LASTEXITCODE -ne 0) { throw 'Regression tests failed' }
    }
    if (-not $BuildOnly) {
        $runArguments=@('--project-root',$projectRoot,'--mode',$Mode)
        if ($null -ne $Samples) { $runArguments+=@('--samples',$Samples.ToString()) }
        if ($null -ne $BlockSize) { $runArguments+=@('--block-size',$BlockSize.ToString()) }
        if ($null -ne $Seed) { $runArguments+=@('--seed',$Seed.ToString()) }
        if ($null -ne $Repeats) { $runArguments+=@('--repeats',$Repeats.ToString()) }
        if ($null -ne $Warmup) { $runArguments+=@('--warmup',$Warmup.ToString()) }
        if ($OutputDirectory) { $runArguments+=@('--output-dir',$OutputDirectory) }
        & (Join-Path $buildRoot 'portfolio_cuda.exe') @runArguments
        if ($LASTEXITCODE -ne 0) { throw "Application failed with exit code $LASTEXITCODE" }
    }
} finally {
    $env:TEMP=$previousTemp;$env:TMP=$previousTmp
}
