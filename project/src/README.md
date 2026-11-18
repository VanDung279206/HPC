# A04 Monte Carlo tối ưu danh mục trên GPU

Phần A04 gồm hai bản CUDA dùng cùng dữ liệu, phân phối trọng số, seed và phép reduction. Bản naive đọc μ và Σ từ Global Memory; bản optimized đưa hai mảng vào Shared Memory theo kích thước K thực tế. Cả hai tìm danh mục có Sharpe lớn nhất và danh mục rủi ro nhỏ nhất thỏa R_min trong M mẫu đã sinh.

## Build và chạy trên Windows

Cần NVIDIA GPU, CUDA Toolkit, Visual Studio C++ Build Tools, CMake và Ninja. Script dùng vswhere để tìm Visual Studio, ưu tiên CMake/Ninja trên PATH rồi tới bản đi kèm Visual Studio. CUDA Toolkit được CMake tìm qua môi trường/cài đặt tiêu chuẩn. Không có đường dẫn cố định tới phiên bản MSVC hoặc CUDA.

Từ thư mục gốc repository:

~~~powershell
.\project\src\cuda\build_and_run.ps1 -RunTests
~~~

Lệnh này build Release, chạy kiểm thử rồi chạy verify và Monte Carlo với cấu hình chung. GPU mặc định là sm_75, phù hợp Quadro RTX 3000. Dùng tham số Architecture khi chạy GPU khác:

~~~powershell
.\project\src\cuda\build_and_run.ps1 -Architecture 75 -BuildOnly -RunTests
.\project\src\cuda\build_and_run.ps1 -Mode mc -Samples 1000000 -BlockSize 256 -Repeats 5
~~~

Script chạy được từ thư mục bất kỳ vì xác định project từ vị trí của chính script. BuildDirectory cho phép chọn nơi build riêng. Không dùng lại binary Debug cũ trong out/build/x64-Debug để benchmark.

Chạy binary sau khi build:

~~~powershell
& .\project\out\build\cuda-release\portfolio_cuda.exe --project-root .\project --mode verify
& .\project\out\build\cuda-release\portfolio_cuda.exe --project-root .\project --mode mc --samples 10000000 --block-size 128 --seed 42 --warmup 1 --repeats 5
~~~

Chương trình kết thúc ngay khi chạy xong; chỉ chờ Enter khi truyền --pause. --block_size cũ vẫn được hỗ trợ như bí danh của --block-size.

Build bằng CMake trên môi trường đã cấu hình compiler:

~~~powershell
cmake -S project -B project/out/build/cuda-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TRY_COMPILE_CONFIGURATION=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build project/out/build/cuda-release
ctest --test-dir project/out/build/cuda-release --output-on-failure
~~~

Có thể dùng -S . thay cho -S project để build từ CMake gốc repository.

Fast math mặc định tắt. Muốn thử fast math, dùng -FastMath trong script hoặc -DPORTFOLIO_FAST_MATH=ON với CMake, chạy lại kiểm thử và ghi rõ cấu hình trong báo cáo.

## Cấu hình và dữ liệu

Chương trình đọc project/config/run_config.json trước; tham số dòng lệnh ghi đè giá trị trong JSON. --config nhận đường dẫn tới file cấu hình khác. Khi truyền --project-root, chương trình chỉ lấy dữ liệu từ project đó, không tìm file thay thế ở thư mục khác.

| JSON | Dòng lệnh | Ý nghĩa |
|---|---|---|
| K | Lấy từ JSON và đối chiếu mu.txt | Số tài sản, từ 1 đến 64 |
| M | --samples | Số danh mục, phải dương |
| Rf | --rf | Lợi suất phi rủi ro |
| R_min | --r-min | Lợi suất tối thiểu cho Min Risk |
| seed | --seed | Seed uint64 không âm |
| block_size | --block-size | Chỉ nhận 64, 128, 256, 512 |
| cloud_sample_size | --cloud-size | Số điểm minh họa, không âm |
| warmup_runs | --warmup | Số lượt khởi động mỗi phương pháp, không âm |
| repeats | --repeats | Số lượt đo mỗi phương pháp, ít nhất 1 |

JSON là một object phẳng chứa các trường số trên; trường nguyên cần ghi dưới dạng số nguyên. Trường sai tên, trùng khóa, cấu trúc không hỗ trợ và số không hữu hạn đều bị từ chối.

Dữ liệu đầu vào:

- data/processed/mu.txt: K, sau đó đúng K số.
- data/processed/cov.txt: đúng K×K số theo thứ tự hàng.
- data/processed/tickers.txt: đúng K ticker khác nhau.
- tests/test_data/test_weights.txt: N K, sau đó đúng N×K trọng số.

Dữ liệu thiếu/dư, NaN/Inf, covariance không đối xứng/không nửa xác định dương, trọng số âm hoặc không có tổng gần 1 đều gây lỗi. Kiểm tra covariance có dung sai theo độ lớn của ma trận để chấp nhận sai số float32. Một ma trận covariance bằng 0 có đầy đủ dữ liệu được phép; file covariance rỗng bị từ chối.

μ, Σ, Rf và R_min phải dùng cùng kỳ thời gian. Pipeline hiện tại quy đổi μ và Σ theo năm, vì vậy Rf=0.04 và R_min=0.12 là các giá trị theo năm.

## Thuật toán và tính tái lập

Mỗi sample_id dùng Philox với (seed, sample_id, offset=0). Sinh K biến Exp(1) bằng -log(U), sau đó chia cho tổng để được Dirichlet(1,…,1). Khi tổng bằng 0 do trường hợp biên số học, dùng trọng số bằng nhau.

Cách lấy mẫu này thống nhất về phân phối với bản CPU dùng -log(U). CPU mt19937 và GPU Philox không tạo cùng chuỗi trọng số chỉ vì cùng seed; dùng bộ trọng số cố định để so sánh số học.

Công thức:

- Rp = tổng w_i μ_i.
- σp = sqrt(wᵀΣw).
- Sharpe = (Rp − Rf)/σp.
- Khi σp ≤ 1e-8, Sharpe dùng giá trị quy ước -1e9; CPU tham chiếu dùng cùng quy ước.
- Min Risk chỉ xét các danh mục có Rp ≥ R_min.

Reduction theo cây chạy trong từng block; sau đó CPU chọn kết quả cuối từ các block. Block cuối có thể thiếu mẫu nhưng toàn bộ thread vẫn tham gia barrier. Naive và optimized đều có reduction, vì vậy phần khác biệt đang đo là nơi lưu μ, Σ.

Với K=10, có nhánh biên dịch chuyên biệt để tránh mảng trọng số cục bộ dư cho 64 tài sản. K khác dùng nhánh tổng quát tới 64. Shared Memory optimized được cấp theo (K + K²)×4 + block_size×20 byte; naive chỉ cần block_size×20 byte.

## Verify

Mode verify và all kiểm tra cả naive và optimized trên cùng trọng số cố định. CPU tham chiếu tích lũy bằng double từ các đầu vào float32 mà GPU nhận. Mỗi return, risk và Sharpe phải hữu hạn và có sai số tuyệt đối < 1e-5.

Mã thoát:

- 0: toàn bộ công việc được yêu cầu thành công.
- 1: cấu hình, dữ liệu, I/O hoặc CUDA lỗi.
- 2: đối chiếu số học không đạt.

Mode all dừng trước benchmark nếu verify không đạt. Thiếu file trọng số không được coi là verify thành công.

Các kiểm thử CTest bao gồm đối chiếu số học, K=1/3/10/64, block 64/128/256/512, block cuối thiếu mẫu, M nhỏ, không có nghiệm đạt R_min, seed tái lập và thống kê Dirichlet. Bộ CLI kiểm tra đầu vào lỗi, cấu hình, mã thoát và đầu ra CSV. Máy không có GPU có thể báo SKIP cho các kiểm thử cần CUDA.

## Benchmark

Hai bản đều tắt ghi cloud trong khoảng benchmark. Mỗi bản có lượt khởi động; các lượt đo tiếp theo đổi thứ tự naive/optimized luân phiên. Báo cáo median của các lượt đo và giữ từng lượt trong CSV.

| Khoảng đo | Đồng hồ | Nội dung |
|---|---|---|
| allocation_ms | CPU steady_clock | Cấp phát host/device, tạo event và khởi tạo bộ đếm |
| H2D_ms | CUDA Events | Sao chép μ và Σ |
| kernel_ms | CUDA Events | cuRAND, sinh trọng số, tính chỉ số và reduction theo block |
| D2H_ms | CUDA Events | Sao chép kết quả block và số mẫu hợp lệ |
| host_reduce_ms | CPU steady_clock | Chọn nghiệm tốt nhất toàn cục |
| cleanup_ms | CPU steady_clock | Giải phóng các buffer device |
| total_ms | CPU steady_clock | Toàn bộ wrapper từ trước cấp phát tới sau hủy buffer/event |

total_ms được đo trực tiếp, không phải tổng các cột thành phần. Nó gồm cả các khoảng điều phối và hủy đối tượng không được tách thành cột riêng. Kiểm tra tính hợp lệ của tham số/mô hình và truy vấn giới hạn launch được thực hiện trước khoảng đo này. Đọc dữ liệu từ đĩa và ghi CSV nằm ngoài benchmark.

Cloud được sinh trong lượt riêng, chỉ cho min(M, cloud_sample_size) sample_id đầu tiên, không chạy lại toàn bộ M khi chỉ cần vài nghìn điểm.

Kết quả có Sharpe tốt nhất chỉ là nghiệm tốt nhất trong mẫu Monte Carlo. Cloud là đám mây danh mục để minh họa; không tự động được gọi là đường biên hiệu quả chính xác.

## Đầu ra

Mặc định ghi vào project/results/gpu; dùng --output-dir để chọn thư mục khác. Đường dẫn tương đối tính từ project root. CSV của GPU tách khỏi thư mục kết quả CPU.

| File | Nội dung |
|---|---|
| run_metadata.txt | GPU, compute capability, driver/runtime, compiler, fast math, seed và cấu hình |
| tables/verify_gpu.csv | Trọng số, kết quả double CPU, kết quả hai GPU và PASS từng mẫu |
| tables/benchmark_times.csv | Từng lượt đo, các pha thời gian và speedup |
| tables/benchmark_summary.csv | Median từng cột; repeat=0 biểu thị dòng tổng hợp |
| tables/optimal_portfolios.csv | Max Sharpe và Min Risk; thời gian là median end-to-end |
| figures/cloud_points.csv | sample_id, chỉ số và trọng số theo tên ticker |

Khi không có mẫu đạt R_min, console thông báo rõ và CSV không ghi dòng Min_Risk_Rmin. Khi cloud size bằng 0, file cloud cũ trong cùng thư mục output được bỏ để tránh dùng nhầm dữ liệu.

Các file được ghi lại cho mỗi lần chạy. Dùng output-dir khác nhau để lưu riêng các cấu hình M/block/seed. Khi lập biểu đồ speedup so với CPU, cần ghép thời gian CPU tuần tự/OpenMP cùng cấu hình vào các bảng GPU; speedup trong chương trình này hiện là optimized so với naive.
