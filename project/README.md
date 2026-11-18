# A03/A05/A06 — Portfolio Monte Carlo (CPU + GPU)

Tối ưu hóa danh mục đầu tư bằng Monte Carlo trên CPU (tuần tự + OpenMP) và GPU (CUDA).  
Cấu hình đọc từ **`config/run_config.json`** — không cần biên dịch lại khi đổi M, seed, Rf, R_min.

---

## Cấu trúc thư mục

```
project/
├── config/
│   └── run_config.json         ← Cấu hình chung (M, seed, Rf, R_min, K…)
├── src/
│   ├── cpu/
│   │   ├── cpu_mc.cpp          ← CPU Sequential + OpenMP
│   │   └── gen_outputs.py
│   └── gpu/
│       └── portfolio_cuda.cu   ← GPU Naive + Optimized (cần CUDA Toolkit)
├── data/processed/
│   ├── mu.txt                  ← Vector lợi suất kỳ vọng (10 cổ phiếu, năm hoá)
│   ├── cov.txt                 ← Ma trận hiệp phương sai 10×10 (năm hoá)
│   └── tickers.txt             ← Tên 10 mã cổ phiếu
├── tests/test_data/
│   └── test_weights.txt        ← 10.000 bộ trọng số cố định (đối chiếu CPU/GPU)
├── results/tables/             ← Kết quả đầu ra (tự tạo khi chạy)
│   ├── verify_cpu.csv          ← 10.000 mẫu: return/risk/sharpe (CPU)
│   ├── verify_gpu.csv          ← 10.000 mẫu: return/risk/sharpe (GPU)
│   ├── cloud_points.csv        ← 3.000 điểm frontier (from sequential)
│   ├── optimal_portfolios.csv  ← Bảng tổng hợp 4 phương pháp (cột thống nhất)
│   └── benchmark_cpu.csv       ← Thời gian + speedup CPU
├── CMakeLists.txt              ← Build CPU-only hoặc CPU+GPU
├── build_msvc_auto.bat         ← Build + run MSVC (CPU-only, dùng cái này!)
├── build_msvc.bat              ← Build MSVC (Developer Prompt)
└── build_mingw.bat             ← Build MinGW/GCC
```

---

## Cấu hình (`config/run_config.json`)

```json
{
  "K": 10,
  "M": 1000000,
  "Rf": 0.04,
  "R_min": 0.12,
  "seed": 42,
  "block_size": 256,
  "cloud_sample_size": 2000
}
```

| Tham số | Mô tả | Giá trị theo đề cương |
|---------|-------|----------------------|
| `M` | Số kịch bản Monte Carlo | 100 000 / 1 000 000 / 10 000 000 / 100 000 000 |
| `seed` | Random seed (cố định để tái lập) | **42** |
| `Rf` | Lãi suất phi rủi ro (năm hoá) | **0.04** (4%/năm) |
| `R_min` | Lợi suất tối thiểu (năm hoá) | **0.12** (12%/năm) |
| `K` | Số tài sản (phải khớp `mu.txt`) | 10 |

> **Lưu ý:** μ, Σ, Rf và R_min phải cùng đơn vị năm hoá.  
> Chương trình **không** nhân thêm hệ số năm cho Rf hoặc R_min.

---

## Build CPU-only

### MSVC (Khuyến nghị) ⭐

```powershell
Set-Location "E:\HPC\HPC\project"
cmd /c "build_msvc_auto.bat"
```

### CMake — CPU-only (không cần CUDA)

```powershell
cmake -S E:\HPC\HPC\project -B E:\HPC\HPC\project\build -DBUILD_CUDA=OFF
cmake --build E:\HPC\HPC\project\build --config Release
```

### CMake — CPU + GPU (cần CUDA Toolkit)

```powershell
cmake -S E:\HPC\HPC\project -B E:\HPC\HPC\project\build -DBUILD_CUDA=ON
cmake --build E:\HPC\HPC\project\build --config Release
```

### MinGW/GCC (cần MSYS2)

```bash
g++ -std=c++17 -O2 -march=native -fopenmp -Wall src/cpu/cpu_mc.cpp -o cpu_mc.exe
```

---

## Chạy và ghi đè tham số

```powershell
# Chạy với cấu hình trong JSON (mặc định)
cpu_mc.exe "E:\HPC\HPC\project"

# Ghi đè M qua CLI (không cần sửa JSON)
cpu_mc.exe "E:\HPC\HPC\project" --M 10000000

# Ghi đè nhiều tham số
cpu_mc.exe "E:\HPC\HPC\project" --M 100000000 --seed 42 --Rf 0.04 --Rmin 0.12 --threads 16
```

---

## Mốc benchmark M theo đề cương

| M | Ghi chú |
|---|---------|
| 100 000 | Chạy nhanh để kiểm tra |
| 1 000 000 | Cấu hình mặc định trong JSON |
| 10 000 000 | |
| 100 000 000 | Nếu dùng 50 000 000 thay thế phải ghi rõ trong bảng kết quả |

Thay M trong JSON hoặc dùng `--M <giá trị>` khi chạy — **không cần biên dịch lại**.

---

## Đầu ra

| File | Nội dung |
|------|----------|
| `verify_cpu.csv` | `sample_id, w_FPT…w_VPB, return, risk, sharpe` (10.000 mẫu cố định) |
| `verify_gpu.csv` | Tương tự, do GPU tạo — đối chiếu theo `sample_id` |
| `cloud_points.csv` | 3.000 điểm frontier: `sample_id, weights, return, risk, sharpe` |
| `optimal_portfolios.csv` | Bảng tổng hợp 4 phương pháp: `method, criterion, weights, return_Rp, risk_sigma, sharpe_S, M, seed, Rf, R_min, threads, time_ms, speedup` |
| `benchmark_cpu.csv` | Thời gian + speedup: `method, M, seed, Rf, R_min, threads, time_ms, speedup_vs_seq, notes` |

### Phương pháp (`method`) trong `optimal_portfolios.csv`

| Giá trị | Mô tả |
|---------|-------|
| `CPU_Sequential` | CPU tuần tự, 1 luồng |
| `CPU_OpenMP` | CPU đa luồng OpenMP |
| `GPU_Naive` | GPU CUDA kernel naive |
| `GPU_Optimized` | GPU CUDA kernel tối ưu |

---

## Đối chiếu số học CPU/GPU

```
verify_cpu.csv   (CPU)  ─┐
                          ├─ so khớp theo sample_id
verify_gpu.csv   (GPU)  ─┘
```

Sai số chấp nhận được: `|Δ_return|, |Δ_risk|, |Δ_sharpe| < 1e-5`

> **Lưu ý về seed:** cùng `seed=42` với `mt19937` (CPU) và `cuRAND` (GPU)  
> **không** đảm bảo cùng chuỗi số ngẫu nhiên.  
> Đối chiếu số học phải dùng **10.000 trọng số cố định** (`test_weights.txt`).

---

## Phạm vi thời gian đo

| Phương pháp | Bắt đầu | Kết thúc |
|-------------|---------|---------|
| CPU Sequential | Vòng lặp M (sinh trọng số + tính metric) | Kết thúc vòng lặp |
| CPU OpenMP | Bước 2 (tính metric song song) | Kết thúc parallel region |
| GPU | Kernel launch | Kết thúc memcpy D→H |

> Bước sinh trọng số (Step 1) của OpenMP là tuần tự và **không** tính vào `time_ms` của OpenMP  
> → speedup OpenMP chỉ phản ánh gia tốc tính metric. Ghi rõ trong bảng báo cáo.

---

## Thông số kỹ thuật

| Tham số | Giá trị |
|---------|---------|
| Số cổ phiếu (K) | 10 (FPT VCB HPG VIC VNM GAS MSN MWG TCB VPB) |
| Phương pháp sinh trọng số | Dirichlet sampling (−ln U, U ~ Uniform(0,1)) |
| Ràng buộc | wᵢ ≥ 0, \|Σwᵢ − 1\| < 1e-6 |

## Công thức

$$R_p = \sum_{i} w_i \mu_i \qquad \sigma_p = \sqrt{w^T \Sigma w} \qquad S = \frac{R_p - R_f}{\sigma_p}$$

**Danh mục tối ưu:**
- **Max Sharpe:** S lớn nhất trong M kịch bản  
- **Min Risk:** σ_p nhỏ nhất trong các kịch bản thỏa R_p ≥ R_min

---

## Lỗi thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
|-----|-------------|---------|
| `Cannot open config: ...` | Đường dẫn project root sai | Truyền đúng thư mục gốc |
| `Invalid configuration: M must be > 0` | JSON hoặc CLI cho M ≤ 0 | Sửa `run_config.json` hoặc `--M` |
| `K in mu.txt != config K` | File `mu.txt` khác K trong JSON | Kiểm tra dòng đầu `mu.txt` phải là `10` |
| `Cannot open .../mu.txt` | Sai đường dẫn | `cpu_mc.exe "E:\HPC\HPC\project"` |
| OpenMP không khả dụng | Compile thiếu `/openmp` | Dùng `build_msvc_auto.bat` |
