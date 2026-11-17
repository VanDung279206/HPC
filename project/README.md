# A03 — CPU Monte Carlo Portfolio Optimization

Chương trình tối ưu hóa danh mục đầu tư bằng phương pháp Monte Carlo trên CPU,  
gồm 3 chế độ: **Kiểm thử tĩnh**, **Tuần tự (Sequential)** và **Đa luồng (OpenMP)**.

---

## Yêu cầu hệ thống

| Thành phần | Yêu cầu |
|------------|---------|
| Hệ điều hành | Windows 10/11 (64-bit) |
| Compiler | **Visual Studio 2019/2022/2026** (có C++ workload) HOẶC MinGW-w64 (MSYS2) |
| CPU | Hỗ trợ OpenMP (hầu hết CPU hiện đại) |
| RAM | Tối thiểu 512 MB |

> Nếu chưa có Visual Studio, tải miễn phí tại:  
> https://visualstudio.microsoft.com/downloads/ → chọn **Community** → tích **Desktop development with C++**

---

## Cấu trúc thư mục

```
project/
├── src/cpu/
│   └── cpu_mc.cpp              ← Mã nguồn chính
├── data/processed/
│   ├── mu.txt                  ← Vector lợi suất kỳ vọng (10 cổ phiếu)
│   ├── cov.txt                 ← Ma trận hiệp phương sai 10×10
│   └── tickers.txt             ← Tên 10 mã cổ phiếu
├── tests/test_data/
│   └── test_weights.txt        ← 10.000 bộ trọng số cố định
├── results/tables/             ← Kết quả đầu ra (tự tạo khi chạy)
│   ├── verify_cpu.csv
│   ├── cloud_points.csv
│   └── optimal_portfolios.csv
├── build_msvc_auto.bat         ← Script build + run (MSVC, dùng cái này!)
├── build_msvc.bat              ← Script build MSVC (cần Developer Prompt)
└── build_mingw.bat             ← Script build MinGW/GCC
```

---

## Hướng dẫn chạy

### ✅ Cách 1 — PowerShell (Khuyến nghị) ⭐

Mở **PowerShell** (`Win + X` → chọn `Terminal` hoặc `Windows PowerShell`), gõ:

**Lần đầu (compile + chạy):**
```powershell
Set-Location "E:\HPC\HPC\project"
cmd /c "build_msvc_auto.bat"
```

**Những lần sau (đã có exe, chạy lại không cần compile):**
```powershell
& "E:\HPC\HPC\project\cpu_mc.exe" "E:\HPC\HPC\project"
```

> Script `build_msvc_auto.bat` tự tìm Visual Studio và compile — không cần cấu hình thêm.

---

### ✅ Cách 2 — Command Prompt (CMD)

Mở **CMD** (`Win + R` → gõ `cmd` → Enter):

```bat
cd /d E:\HPC\HPC\project
build_msvc_auto.bat
```

---

### ✅ Cách 3 — Dùng Developer Command Prompt

1. Mở **Start** → tìm `Developer Command Prompt for VS`
2. Gõ:

```bat
cd /d E:\HPC\HPC\project
build_msvc.bat
```

---

### ✅ Cách 4 — Dùng MinGW/GCC (cần cài MSYS2)

> Bỏ qua nếu đã dùng Visual Studio

**Bước 1:** Cài MSYS2 từ https://www.msys2.org/

**Bước 2:** Mở MSYS2 terminal, cài GCC + OpenMP:
```bash
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-openmp
```

**Bước 3:** Chạy trong MSYS2 terminal:
```bash
cd /e/HPC/HPC/project
g++ -std=c++17 -O2 -march=native -fopenmp -Wall src/cpu/cpu_mc.cpp -o cpu_mc.exe
./cpu_mc.exe "E:/HPC/HPC/project"
```

---

## Kết quả mong đợi

Sau khi chạy thành công, màn hình hiển thị:

```
Portfolio MC Optimizer (CPU)
  Assets : FPT VCB HPG VIC VNM GAS MSN MWG TCB VPB 
  K=10  Rf=0.04  R_min=0.12

=== STATIC ARITHMETIC TEST (verify_cpu.csv) ===
  Loaded 10000 fixed weight sets
  Valid samples written : 10000
  Skipped (bad weights) : 0
  Output -> .../results/tables/verify_cpu.csv

=== SEQUENTIAL CPU Monte Carlo ===
  M = 1000000 scenarios, seed = 42
  Time  : ~205 ms
  Max Sharpe : S=2.05 Rp=0.67 sigma=0.31
  Min Risk   : sigma=0.17 Rp=0.15 S=0.64

=== OPENMP CPU Monte Carlo ===
  M = 1000000 scenarios, threads = 16, seed = 42 (reproducible)
  [1/2] Generating 1000000 weight vectors (sequential RNG)...
  [2/2] Computing metrics in parallel...
  Time  : ~16 ms
  Max Sharpe : S=2.05 Rp=0.67 sigma=0.31
  Min Risk   : sigma=0.17 Rp=0.15 S=0.64

  Speedup_OpenMP = ~12.6x
  Cloud points (3000) -> .../results/tables/cloud_points.csv
  Optimal portfolios -> .../results/tables/optimal_portfolios.csv

All outputs written to .../results/tables
```

---

## File kết quả

| File | Nội dung | Dùng để |
|------|----------|---------|
| `verify_cpu.csv` | 10.000 mẫu: `sample_id, w_FPT…w_VPB, return, risk, sharpe` | Đối chiếu với GPU (Task A05) |
| `cloud_points.csv` | 3.000 điểm: `sample_id, weights, return, risk, sharpe` | Vẽ Đường biên hiệu quả |
| `optimal_portfolios.csv` | 4 danh mục tối ưu (seq/omp × Sharpe/Risk) + thời gian + speedup | Báo cáo kết quả |

---

## Hướng dẫn đối chiếu với GPU (Cross-validation Task A03 & Task A05)

1. **Nguyên lý kiểm thử tĩnh (Static Verification):**
   - File `tests/test_data/test_weights.txt` chứa $10.000$ bộ trọng số cố định ($W$) đã chuẩn hóa $\sum w_i = 1, w_i \ge 0$.
   - Khi chạy `cpu_mc.exe`, chương trình tự động tính toán số học trên $10.000$ bộ trọng số này bằng CPU và xuất ra `results/tables/verify_cpu.csv`.
   - Phía GPU (CUDA kernel ở Task A05) cũng đọc cùng file `test_weights.txt` và xuất ra `verify_gpu.csv`.

2. **Cách đối chiếu:**
   - So sánh từng hàng giữa `verify_cpu.csv` và `verify_gpu.csv`:
     $$\Delta_{\text{return}} = |R_{p,\text{CPU}} - R_{p,\text{GPU}}| < 10^{-5}$$
     $$\Delta_{\text{risk}} = |\sigma_{p,\text{CPU}} - \sigma_{p,\text{GPU}}| < 10^{-5}$$
     $$\Delta_{\text{sharpe}} = |S_{\text{CPU}} - S_{\text{GPU}}| < 10^{-5}$$
   - Nếu toàn bộ $10.000$ mẫu đều có sai số $< 10^{-5}$, kernel tính toán trên GPU được xác nhận là hoàn toàn chính xác về mặt số học.

---

## Thông số kỹ thuật

| Tham số | Giá trị |
|---------|---------|
| Số cổ phiếu (K) | 10 (FPT VCB HPG VIC VNM GAS MSN MWG TCB VPB) |
| Số kịch bản Monte Carlo (M) | 1.000.000 |
| Số mẫu kiểm thử tĩnh | 10.000 |
| Số điểm đám mây xuất ra | 3.000 |
| Lãi suất phi rủi ro (Rf) | 4% |
| Lợi suất tối thiểu (R_min) | 12% |
| Random seed | 42 |
| Phương pháp sinh trọng số | Dirichlet sampling (−ln U, U ~ Uniform(0,1)) |
| Ràng buộc | wᵢ ≥ 0, \|Σwᵢ − 1\| < 10⁻⁶ |

---

## Công thức tính

$$R_p = \sum_{i} w_i \mu_i$$

$$\sigma_p = \sqrt{w^T \Sigma w}$$

$$S = \frac{R_p - R_f}{\sigma_p}$$

**Danh mục tối ưu:**
- **Max Sharpe:** S lớn nhất trong M kịch bản  
- **Min Risk:** σ_p nhỏ nhất trong các kịch bản thỏa R_p ≥ 12%

**Speedup đa luồng:**

$$\text{Speedup\_OpenMP} = \frac{T_{\text{sequential}}}{T_{\text{OpenMP}}}$$

---

## Lỗi thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
|-----|-------------|---------|
| Chạy xong không in gì | Truyền đường dẫn có dấu `\` cuối | Dùng `build_msvc_auto.bat` (đã sửa) hoặc bỏ `\` cuối khi truyền tay |
| `[ERROR] Khong tim thay g++` | Chưa cài MSYS2 hoặc sai đường dẫn | Dùng `build_msvc_auto.bat` thay thế |
| `K in mu.txt != compiled K` | File `mu.txt` bị thay đổi | Kiểm tra dòng đầu `mu.txt` phải là `10` |
| `Cannot open .../mu.txt` | Sai đường dẫn project root | Truyền đúng thư mục gốc: `cpu_mc.exe "E:\HPC\HPC\project"` |
| Lỗi DLL Dev-Cpp | Phiên bản Dev-Cpp cũ, xung đột DLL | Dùng Visual Studio hoặc cài lại MSYS2 |

---

