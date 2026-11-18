param([Parameter(Mandatory=$true)][string]$Executable,
      [Parameter(Mandatory=$true)][string]$ProjectRoot)
$ErrorActionPreference='Stop'
$Executable=(Resolve-Path -LiteralPath $Executable).Path
$ProjectRoot=(Resolve-Path -LiteralPath $ProjectRoot).Path
$tempParent=[IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$fixture=Join-Path $tempParent ('hpc-a04-cli-'+[Guid]::NewGuid().ToString('N'))
$script:passed=0
function Reset-Inputs {
    $data=Join-Path $fixture 'data/processed'
    $tests=Join-Path $fixture 'tests/test_data'
    $config=Join-Path $fixture 'config'
    New-Item -ItemType Directory -Path $data,$tests,$config -Force | Out-Null
    foreach ($name in @('mu.txt','cov.txt','tickers.txt')) {
        Copy-Item -LiteralPath (Join-Path $ProjectRoot "data/processed/$name") -Destination $data -Force
    }
    Copy-Item -LiteralPath (Join-Path $ProjectRoot 'tests/test_data/test_weights.txt') -Destination $tests -Force
    Copy-Item -LiteralPath (Join-Path $ProjectRoot 'config/run_config.json') -Destination $config -Force
}
function Invoke-Case([string]$Name,[int]$Expected,[string[]]$Extra) {
    $ErrorActionPreference='Continue'
    $runArguments=@('--project-root',$fixture,'--output-dir',(Join-Path $fixture "results/$Name"),'--warmup','0')
    if ($Extra -notcontains '--repeats') { $runArguments+=@('--repeats','1') }
    $text=(& $Executable @runArguments @Extra 2>&1 | Out-String)
    $actual=$LASTEXITCODE
    $ErrorActionPreference='Stop'
    if ($text -match 'no CUDA-capable device|CUDA driver version is insufficient|no CUDA GPU') {
        Write-Output 'SKIP: CUDA device unavailable'
        exit 77
    }
    if ($actual -ne $Expected) { throw ("$Name expected exit $Expected, got $actual"+[Environment]::NewLine+$text) }
    $script:passed++
    Write-Output "PASS: $Name"
}
function Write-Input([string]$Relative,[string]$Text) {
    [IO.File]::WriteAllText((Join-Path $fixture $Relative),$Text)
}
try {
    Reset-Inputs
    Invoke-Case 'verify' 0 @('--mode','verify')
    $verify=Import-Csv -LiteralPath (Join-Path $fixture 'results/verify/tables/verify_gpu.csv')
    if ($verify.Count -ne 10000 -or ($verify | Where-Object { $_.naive_pass -ne '1' -or $_.optimized_pass -ne '1' })) {
        throw 'Expected 10000 passing verification records'
    }
    Invoke-Case 'small-M' 0 @('--mode','all','--samples','1')
    $cloud=Import-Csv -LiteralPath (Join-Path $fixture 'results/small-M/figures/cloud_points.csv')
    if (@($cloud).Count -ne 1) { throw 'Small M must export exactly one cloud point' }
    $raw=Import-Csv -LiteralPath (Join-Path $fixture 'results/small-M/tables/benchmark_times.csv')
    if (@($raw).Count -ne 2 -or ($raw | Where-Object {$_.M_simulations -ne '1'})) {
        throw 'CLI samples must override config for both variants'
    }
    Invoke-Case 'small-M' 0 @('--mode','mc','--samples','1','--cloud-size','0')
    if (Test-Path -LiteralPath (Join-Path $fixture 'results/small-M/figures/cloud_points.csv')) {
        throw 'Disabled cloud must not leave stale output'
    }
    Invoke-Case 'no-feasible' 0 @('--mode','mc','--samples','65','--r-min','2')
    $optimal=Import-Csv -LiteralPath (Join-Path $fixture 'results/no-feasible/tables/optimal_portfolios.csv')
    if (@($optimal).Count -ne 2 -or ($optimal | Where-Object {$_.criterion -ne 'Max_Sharpe'})) {
        throw 'No-feasible run must omit Min_Risk rows'
    }
    Invoke-Case 'block-192' 1 @('--mode','mc','--block-size','192')
    Invoke-Case 'block-1024' 1 @('--mode','mc','--block-size','1024')
    Invoke-Case 'M-zero' 1 @('--mode','mc','--samples','0')
    Invoke-Case 'M-negative' 1 @('--mode','mc','--samples','-1')
    Invoke-Case 'bad-mode' 1 @('--mode','verfiy')
    Invoke-Case 'bad-argument' 1 @('--sample','1')
    Invoke-Case 'repeats-zero' 1 @('--repeats','0')
    Write-Input 'data/processed/cov.txt' ''
    Invoke-Case 'empty-cov' 1 @('--mode','verify')
    Write-Input 'data/processed/cov.txt' '0.04'
    Invoke-Case 'truncated-cov' 1 @('--mode','verify')
    Reset-Inputs
    $covTokens=((Get-Content -LiteralPath (Join-Path $fixture 'data/processed/cov.txt') -Raw).Trim() -split '\s+')
    $covTokens[1]='0.5'
    Write-Input 'data/processed/cov.txt' ($covTokens -join ' ')
    Invoke-Case 'asymmetric-cov' 1 @('--mode','verify')
    Reset-Inputs
    $covTokens=((Get-Content -LiteralPath (Join-Path $fixture 'data/processed/cov.txt') -Raw).Trim() -split '\s+')
    $covTokens[0]='-1'
    Write-Input 'data/processed/cov.txt' ($covTokens -join ' ')
    Invoke-Case 'negative-cov' 1 @('--mode','verify')
    Reset-Inputs
    Write-Input 'data/processed/mu.txt' (@('10','NaN 0 0 0 0 0 0 0 0 0') -join [Environment]::NewLine)
    Invoke-Case 'nonfinite-mu' 1 @('--mode','verify')
    Reset-Inputs
    Write-Input 'tests/test_data/test_weights.txt' (@('1 1','1') -join [Environment]::NewLine)
    Invoke-Case 'test-K-mismatch' 1 @('--mode','verify')
    Write-Input 'tests/test_data/test_weights.txt' (@('1 10','1') -join [Environment]::NewLine)
    Invoke-Case 'truncated-weights' 1 @('--mode','verify')
    Write-Input 'tests/test_data/test_weights.txt' (@('1 10','0 0 0 0 0 0 0 0 0 0') -join [Environment]::NewLine)
    Invoke-Case 'invalid-weight-sum' 1 @('--mode','verify')
    Remove-Item -LiteralPath (Join-Path $fixture 'tests/test_data/test_weights.txt')
    Invoke-Case 'missing-weights' 1 @('--mode','verify')
    Reset-Inputs
    Write-Input 'data/processed/tickers.txt' 'FPT'
    Invoke-Case 'missing-tickers' 1 @('--mode','verify')
    Reset-Inputs
    Write-Input 'config/run_config.json' '{"K":10,"block_szie":256}'
    Invoke-Case 'unknown-config-key' 1 @('--mode','verify')
    Write-Input 'config/run_config.json' '{"K":10,"K":10}'
    Invoke-Case 'duplicate-config-key' 1 @('--mode','verify')
    Write-Input 'config/run_config.json' '{"K":10,}'
    Invoke-Case 'malformed-config' 1 @('--mode','verify')
    Write-Output "PASS: $script:passed CLI regression cases"
} finally {
    $resolvedFixture=[IO.Path]::GetFullPath($fixture)
    if (-not $resolvedFixture.StartsWith($tempParent,[StringComparison]::OrdinalIgnoreCase) -or
        -not ([IO.Path]::GetFileName($resolvedFixture)).StartsWith('hpc-a04-cli-')) {
        throw 'Refusing to clean an unexpected fixture directory'
    }
    if (Test-Path -LiteralPath $resolvedFixture) { Remove-Item -LiteralPath $resolvedFixture -Recurse -Force }
}
