#!/usr/bin/env python3
"""
compare_results.py
Đối chiếu số học CPU vs GPU (verify_cpu.csv vs verify_gpu.csv)
và tổng hợp bảng optimal_portfolios cho A05/A06.

Usage:
    python compare_results.py <project_root>
    python compare_results.py E:/HPC/HPC/project
"""

import sys
import os
import csv
from pathlib import Path

TOL = 1e-5   # sai số chấp nhận được

def load_csv(path):
    """Return list of dicts from CSV."""
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def compare_verify(cpu_path, gpu_path, tol=TOL):
    """So sánh verify_cpu.csv và verify_gpu.csv theo sample_id."""
    if not Path(gpu_path).exists():
        print(f"[SKIP] GPU verify file not found: {gpu_path}")
        return

    cpu_rows = {r['sample_id']: r for r in load_csv(cpu_path)}
    gpu_rows = {r['sample_id']: r for r in load_csv(gpu_path)}

    # Normalize GPU column names (method/return_Rp/risk_sigma/sharpe_S → return/risk/sharpe)
    def get(row, *keys):
        for k in keys:
            if k in row:
                return float(row[k])
        raise KeyError(f"None of {keys} in row")

    total  = len(cpu_rows)
    passed = 0
    failed = []

    for sid, crow in cpu_rows.items():
        if sid not in gpu_rows:
            failed.append((sid, 'missing_in_gpu', 0, 0, 0))
            continue
        grow = gpu_rows[sid]

        c_ret = get(crow, 'return', 'return_Rp')
        c_risk= get(crow, 'risk',   'risk_sigma')
        c_sh  = get(crow, 'sharpe', 'sharpe_S')

        g_ret = get(grow, 'return', 'return_Rp')
        g_risk= get(grow, 'risk',   'risk_sigma')
        g_sh  = get(grow, 'sharpe', 'sharpe_S')

        d_ret  = abs(c_ret  - g_ret)
        d_risk = abs(c_risk - g_risk)
        d_sh   = abs(c_sh   - g_sh)

        if d_ret < tol and d_risk < tol and d_sh < tol:
            passed += 1
        else:
            failed.append((sid, 'mismatch', d_ret, d_risk, d_sh))

    print(f"\n=== Cross-Validation: CPU vs GPU ===")
    print(f"  Tolerance : {tol:.0e}")
    print(f"  Total     : {total}")
    print(f"  Passed    : {passed}")
    print(f"  Failed    : {len(failed)}")

    if failed:
        print(f"\n  First 10 failures:")
        for item in failed[:10]:
            if item[1] == 'missing_in_gpu':
                print(f"    sample_id={item[0]}  MISSING in GPU output")
            else:
                sid, _, dr, drisk, dsh = item
                print(f"    sample_id={sid}  Δreturn={dr:.2e}  Δrisk={drisk:.2e}  Δsharpe={dsh:.2e}")
    else:
        print(f"\n  ✅ All {total} samples match within tolerance {tol:.0e}")

    return passed == total and len(failed) == 0

def merge_optimal(results_dir):
    """Merge CPU + GPU optimal_portfolios into one summary table."""
    cpu_path = Path(results_dir) / 'optimal_portfolios.csv'
    gpu_path = Path(results_dir) / 'optimal_portfolios_gpu.csv'
    out_path = Path(results_dir) / 'summary_all_methods.csv'

    rows = []
    for path in [cpu_path, gpu_path]:
        if path.exists():
            r = load_csv(path)
            rows.extend(r)
            print(f"  Loaded {len(r)} rows from {path.name}")
        else:
            print(f"  [SKIP] {path.name} not found")

    if not rows:
        print("  No data to merge.")
        return

    # Write merged
    fieldnames = rows[0].keys()
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\n=== Summary Table ===")
    print(f"  Output -> {out_path}")

    # Print readable summary
    methods_seen = set()
    print(f"\n  {'method':<20} {'criterion':<12} {'return_Rp':>10} {'risk_sigma':>10} {'sharpe_S':>10} {'time_ms':>10}")
    print("  " + "-" * 78)
    for r in rows:
        method   = r.get('method', r.get('version', '?'))
        crit     = r.get('criterion', '?')
        ret_val  = float(r.get('return_Rp', r.get('return', 0)))
        risk_val = float(r.get('risk_sigma', r.get('risk', 0)))
        sh_val   = float(r.get('sharpe_S', r.get('sharpe', 0)))
        t_ms     = float(r.get('time_ms', 0))
        print(f"  {method:<20} {crit:<12} {ret_val:>10.6f} {risk_val:>10.6f} {sh_val:>10.6f} {t_ms:>10.2f}")

def main():
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    root = root.rstrip('/\\')
    results_dir = os.path.join(root, 'results', 'tables')

    cpu_verify = os.path.join(results_dir, 'verify_cpu.csv')
    gpu_verify = os.path.join(results_dir, 'verify_gpu.csv')

    if not Path(cpu_verify).exists():
        print(f"[ERROR] CPU verify file not found: {cpu_verify}")
        print("  Run cpu_mc.exe first to generate verify_cpu.csv")
        sys.exit(1)

    compare_verify(cpu_verify, gpu_verify)

    print(f"\n=== Merging Optimal Portfolios ===")
    merge_optimal(results_dir)

if __name__ == '__main__':
    main()
