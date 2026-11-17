"""
gen_outputs.py  –  Bản tham chiếu Python (NumPy) cho Task A03
(Dùng để đối chiếu thuật toán; kết quả C++ / OpenMP thực tế lấy từ cpu_mc.exe)

Output (lưu riêng tại results/python/ để không ghi đè bản C++):
  results/python/verify_cpu.csv       (10 000 mẫu từ test_weights.txt)
  results/python/cloud_points.csv     (3 000 điểm đầu sequential)
  results/python/optimal_portfolios.csv (kết quả tham chiếu NumPy)

Chạy: python gen_outputs.py <project_root>
Yêu cầu: numpy
"""

from __future__ import annotations
import sys, math, time
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("numpy required: pip install numpy")

# ─── Paths ───────────────────────────────────────────────────────
root      = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
DATA      = root / "data" / "processed"
TEST_DATA = root / "tests" / "test_data"
OUT       = root / "results" / "tables"          # C++ outputs (shared)
OUT_PY    = root / "results" / "python"          # Python-only outputs (riêng)
OUT.mkdir(parents=True, exist_ok=True)
OUT_PY.mkdir(parents=True, exist_ok=True)

# ─── Constants ───────────────────────────────────────────────────
RF      = 0.04
R_MIN   = 0.12
SEED    = 42
M       = 1_000_000
N_CLOUD = 3_000
EPS_W   = 1e-6

# ─── Loaders ─────────────────────────────────────────────────────
def load_mu(path):
    toks = Path(path).read_text().split()
    K = int(toks[0])
    mu = np.array([float(x) for x in toks[1:K+1]])
    return K, mu

def load_cov(path, K):
    vals = [float(x) for x in Path(path).read_text().split()]
    return np.array(vals).reshape(K, K)

def load_tickers(path):
    return Path(path).read_text().split()

def load_test_weights(path, K):
    lines = Path(path).read_text().splitlines()
    n, k = map(int, lines[0].split())
    rows = []
    for i in range(1, n+1):
        rows.append(list(map(float, lines[i].split())))
    return np.array(rows)

# ─── Batch metrics ────────────────────────────────────────────────
def batch_metrics(W, mu, cov):
    ret   = W @ mu
    WC    = W @ cov
    var   = np.einsum('ij,ij->i', WC, W)
    risk  = np.sqrt(np.maximum(var, 0.0))
    sharpe = np.where(risk > 1e-12, (ret - RF) / risk, -1e30)
    return ret, risk, sharpe

# ─────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────
K, mu = load_mu(DATA / "mu.txt")
cov   = load_cov(DATA / "cov.txt", K)
tick  = load_tickers(DATA / "tickers.txt")

print(f"Assets: {' '.join(tick)}  K={K}  Rf={RF}  R_min={R_MIN}")

# ─────────────────────────────────────────────────────────────────
# 1. STATIC TEST  →  verify_cpu.csv
# ─────────────────────────────────────────────────────────────────
print("\n=== STATIC ARITHMETIC TEST (verify_cpu.csv) ===")
W_test = load_test_weights(TEST_DATA / "test_weights.txt", K)
N_test = len(W_test)
print(f"  Loaded {N_test} fixed weight sets")

wsum     = W_test.sum(axis=1)
mask_sum = np.abs(wsum - 1.0) < EPS_W
mask_neg = (W_test >= 0.0).all(axis=1)
valid    = mask_sum & mask_neg
bad      = N_test - int(valid.sum())

W_v     = W_test[valid]
ids_v   = np.where(valid)[0]
ret_v, risk_v, sharpe_v = batch_metrics(W_v, mu, cov)

out_path = OUT_PY / "verify_cpu.csv"
with open(out_path, "w") as f:
    hdr = "sample_id," + ",".join(f"w_{t}" for t in tick) + ",return,risk,sharpe"
    f.write(hdr + "\n")
    for i in range(len(ids_v)):
        line = str(ids_v[i])
        for wi in W_v[i]:
            line += f",{wi:.8f}"
        line += f",{ret_v[i]:.8f},{risk_v[i]:.8f},{sharpe_v[i]:.8f}"
        f.write(line + "\n")

print(f"  Valid samples : {int(valid.sum())}")
print(f"  Skipped       : {bad}")
print(f"  Output -> {out_path}")

# ─────────────────────────────────────────────────────────────────
# 2a. SEQUENTIAL MC (seed=42, numpy MT19937)
#
# Pre-generate toàn bộ W_all một lần duy nhất → tái dùng ở section
# 2b (chunked) mà không cần sinh lại, đảm bảo kết quả bit-identical.
# ─────────────────────────────────────────────────────────────────
print(f"\n=== SEQUENTIAL MC  (M={M:,}, seed={SEED}) ===")
print(f"  Pre-generating {M:,} weight vectors (seed={SEED})...")

BATCH = 50_000

# ── Sinh toàn bộ M weights một lần (dùng chung cho cả 2a và 2b) ──
rng_mt = np.random.Generator(np.random.MT19937(seed=SEED))
U_all  = rng_mt.uniform(0.0, 1.0, size=(M, K))
U_all  = np.clip(U_all, 1e-300, 1.0)
E_all  = -np.log(U_all)
W_all  = E_all / E_all.sum(axis=1, keepdims=True)   # shape (M, K)
del U_all, E_all

best_s  = dict(sharpe=-1e30, ret=0.0, risk=0.0, w=None)
best_mr = dict(risk=1e30,  ret=-1.0, sharpe=0.0, w=None)
cloud_data: list[tuple] = []
cloud_full = False

t0 = time.perf_counter()
processed = 0

while processed < M:
    nb = min(BATCH, M - processed)
    W  = W_all[processed : processed + nb]

    ret, risk, sharpe = batch_metrics(W, mu, cov)

    idx_s = int(np.argmax(sharpe))
    if float(sharpe[idx_s]) > best_s["sharpe"]:
        best_s = dict(sharpe=float(sharpe[idx_s]),
                      ret=float(ret[idx_s]),
                      risk=float(risk[idx_s]),
                      w=W[idx_s].tolist())

    mask = ret >= R_MIN
    if mask.any():
        sub  = np.where(mask, risk, 1e30)
        idx_m = int(np.argmin(sub))
        if float(risk[idx_m]) < best_mr["risk"] and float(ret[idx_m]) >= R_MIN:
            best_mr = dict(risk=float(risk[idx_m]),
                           ret=float(ret[idx_m]),
                           sharpe=float(sharpe[idx_m]),
                           w=W[idx_m].tolist())

    if not cloud_full:
        need = N_CLOUD - len(cloud_data)
        take = min(need, nb)
        for i in range(take):
            cloud_data.append((processed+i, W[i].tolist(),
                               float(ret[i]), float(risk[i]), float(sharpe[i])))
        if len(cloud_data) >= N_CLOUD:
            cloud_full = True

    processed += nb

t1 = time.perf_counter()
seq_ms = (t1 - t0) * 1000
print(f"  Time  : {seq_ms:.1f} ms")
print(f"  Max Sharpe : S={best_s['sharpe']:.6f}  Rp={best_s['ret']:.6f}  σ={best_s['risk']:.6f}")
if best_mr["ret"] >= 0:
    print(f"  Min Risk   : σ={best_mr['risk']:.6f}  Rp={best_mr['ret']:.6f}  S={best_mr['sharpe']:.6f}")
else:
    print(f"  Min Risk   : No portfolio found (Rp >= {R_MIN})")

# ─────────────────────────────────────────────────────────────────
# 2b. NumPy CHUNKED – bản tham chiếu, chạy tuần tự trên Python
#
# Mục đích: kiểm tra tính đúng đắn của việc chia chunk (giống cách
#   OpenMP chia work), KHÔNG phải đo speedup OpenMP.
#   → Số liệu speedup OpenMP thực tế lấy từ cpu_mc.exe (C++).
#
# Reproducibility: dùng lại W_all đã sinh ở trên (cùng seed=42)
#   → kết quả chunked phải khớp hoàn toàn với sequential.
# ─────────────────────────────────────────────────────────────────
N_THREADS = 16
chunk_m   = M // N_THREADS   # số scenario mỗi chunk (trừ chunk cuối)

print(f"\n=== NUMPY CHUNKED REFERENCE  (N_CHUNKS={N_THREADS}, M={M:,}) ===")
print(f"  (Chay tuan tu Python – khong phai OpenMP that su)")

cref_bs = dict(sharpe=-1e30, ret=0.0, risk=0.0, w=None)
cref_mr = dict(risk=1e30,  ret=-1.0, sharpe=0.0, w=None)

# Dùng lại W_all đã pre-generate ở section 2a (seed=42, không sinh lại)
t0 = time.perf_counter()

for tid in range(N_THREADS):
    start   = tid * chunk_m
    m_chunk = M - chunk_m * (N_THREADS - 1) if tid == N_THREADS - 1 else chunk_m

    done = 0
    while done < m_chunk:
        nb = min(BATCH, m_chunk - done)
        W  = W_all[start + done : start + done + nb]

        ret, risk, sharpe = batch_metrics(W, mu, cov)

        idx_s = int(np.argmax(sharpe))
        if float(sharpe[idx_s]) > cref_bs["sharpe"]:
            cref_bs = dict(sharpe=float(sharpe[idx_s]),
                           ret=float(ret[idx_s]),
                           risk=float(risk[idx_s]),
                           w=W[idx_s].tolist())

        mask = ret >= R_MIN
        if mask.any():
            sub   = np.where(mask, risk, 1e30)
            idx_m = int(np.argmin(sub))
            if float(risk[idx_m]) < cref_mr["risk"] and float(ret[idx_m]) >= R_MIN:
                cref_mr = dict(risk=float(risk[idx_m]),
                               ret=float(ret[idx_m]),
                               sharpe=float(sharpe[idx_m]),
                               w=W[idx_m].tolist())
        done += nb

t1 = time.perf_counter()
cref_ms = (t1 - t0) * 1000

print(f"  Time       : {cref_ms:.1f} ms  (Python sequential)")
print(f"  Max Sharpe : S={cref_bs['sharpe']:.6f}  Rp={cref_bs['ret']:.6f}  σ={cref_bs['risk']:.6f}")
if cref_mr["ret"] >= 0:
    print(f"  Min Risk   : σ={cref_mr['risk']:.6f}  Rp={cref_mr['ret']:.6f}  S={cref_mr['sharpe']:.6f}")
    cref_mr_valid = True
else:
    print(f"  Min Risk   : No portfolio found (Rp >= {R_MIN})")
    cref_mr_valid = False

# Kiểm tra tự động: chunked phải cho kết quả giống sequential
ok_sharpe = abs(cref_bs["sharpe"] - best_s["sharpe"]) < 1e-10
ok_risk   = abs(cref_mr["risk"]   - best_mr["risk"])  < 1e-10 if cref_mr_valid else True
print(f"  Kiem tra khop sequential : Sharpe={'OK' if ok_sharpe else 'SAI'}"
      f"  MinRisk={'OK' if ok_risk else 'SAI'}")

# ─────────────────────────────────────────────────────────────────
# Write cloud_points.csv  →  results/python/  (không ghi đè C++)
# ─────────────────────────────────────────────────────────────────
cloud_path = OUT_PY / "cloud_points.csv"
with open(cloud_path, "w") as f:
    hdr = "sample_id," + ",".join(f"w_{t}" for t in tick) + ",return,risk,sharpe"
    f.write(hdr + "\n")
    for m_id, w_list, ret_v2, risk_v2, sh_v2 in cloud_data:
        line = str(m_id)
        for wi in w_list:
            line += f",{wi:.8f}"
        line += f",{ret_v2:.8f},{risk_v2:.8f},{sh_v2:.8f}"
        f.write(line + "\n")
print(f"\n  Cloud points ({len(cloud_data)}) -> {cloud_path}")

# ─────────────────────────────────────────────────────────────────
# Write optimal_portfolios.csv  →  results/python/
# Chỉ ghi kết quả Python sequential (không có hàng "openmp" giả)
# Khi có kết quả C++ thực tế, optimal_portfolios.csv ở results/tables/
# sẽ được ghi bởi cpu_mc.exe với số liệu OpenMP chính xác.
# ─────────────────────────────────────────────────────────────────
opt_path = OUT_PY / "optimal_portfolios.csv"
with open(opt_path, "w") as f:
    hdr = ("version,criterion,"
           + ",".join(f"w_{t}" for t in tick)
           + ",return,risk,sharpe,time_ms")
    f.write(hdr + "\n")

    def write_row(ver, crit, p, tms):
        line = f"{ver},{crit}"
        for wi in p["w"]:
            line += f",{wi:.8f}"
        line += (f",{p['ret']:.8f},{p['risk']:.8f},{p['sharpe']:.8f}"
                 f",{tms:.2f}")
        f.write(line + "\n")

    write_row("numpy_sequential", "max_sharpe", best_s,  seq_ms)
    if best_mr["ret"] >= 0:
        write_row("numpy_sequential", "min_risk",   best_mr, seq_ms)

print(f"  Optimal portfolios -> {opt_path}")
print(f"\n{'='*60}")
print(f"  Python sequential time = {seq_ms:.1f} ms")
print(f"  → Chay cpu_mc.exe de lay ket qua OpenMP thuc te (C++)!")
print(f"{'='*60}")
