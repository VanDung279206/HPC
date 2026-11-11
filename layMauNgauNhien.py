from pathlib import Path
import numpy as np

# 1. Cố định seed = 42 để đảm bảo tính tái lập (reproducibility)
rng = np.random.default_rng(42)

num_samples = 10000  # 10.000 mẫu kiểm thử
K = 10               # 10 mã cổ phiếu VN30

# 2. Sinh ngẫu nhiên phân phối đều [0, 1] dạng float32
raw_u = rng.uniform(0.0, 1.0, size=(num_samples, K)).astype(np.float32)

# 3. Chuẩn hóa để tổng mỗi hàng bằng 1.0 (không bán khống, dùng hết 100% vốn)
test_weights = raw_u / raw_u.sum(axis=1, keepdims=True)

# 4. Xác định đường dẫn đích và tự động tạo thư mục nếu chưa tồn tại
output_path = Path("project/tests/test_data/test_weights.txt")
output_path.parent.mkdir(parents=True, exist_ok=True)

# 5. Ghi tệp test_weights.txt (Dòng 1: Số mẫu và K; các dòng sau: bộ tỷ trọng)
with output_path.open("w", encoding="utf-8") as f:
    f.write(f"{num_samples} {K}\n")
    for row in test_weights:
        f.write(" ".join(f"{val:.8f}" for val in row) + "\n")

print(f"Đã cập nhật thành công tệp {output_path} với {num_samples} bộ trọng số!")