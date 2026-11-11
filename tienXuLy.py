'''
1. đọc dữ liệu từ file CSV
2. kiểm tra và làm sạch dữ liệu
3. tính lợi suất cố phiếu hằng ngày
4. tính lợi suất kỳ vọng và ma trận hiệp phương sai
5. xuất dữ liệu thành file .txt
'''

from __future__ import annotations

from pathlib import Path
from tkinter import Tk, filedialog, messagebox

import numpy as np
import pandas as pd

# khai báo các hằng số
TICKERS = [
    "FPT", "VCB", "HPG", "VIC", "VNM",
    "GAS", "MSN", "MWG", "TCB", "VPB"
]

TRADING_DAYS_PER_YEAR = 252 # số phiên giao dịch trong 1 năm

# chọn file và thư mục đầu vào
def select_input_file() -> Path | None:
    root = Tk()
    root.withdraw()

    selected_file = filedialog.askopenfilename(
        title="Chọn file để xử lý",
        filetypes=[
            ("CSV Files", "*.csv"),
            ("All Files", "*.*")
        ]
    )

    root.destroy()

    if not selected_file:
        return None

    return Path(selected_file)

# hàm lưu thư mục đầu ra
def select_output_folder() -> Path | None:
    root = Tk()
    root.withdraw()

    selected_folder = filedialog.askdirectory(
        title="Chọn thư mục lưu kết quả"
    )

    root.destroy()

    if not selected_folder:
        return None

    return Path(selected_folder)

# kiểm tra dữ liệu và load dữ liệu từ file CSV
def load_prices(input_file: Path) -> pd.DataFrame:

    if not input_file.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {input_file}"
        )

    prices = pd.read_csv(
        input_file,
        parse_dates=["Date"]
    )

    required = ["Date", *TICKERS]

    missing_columns = [
        column for column in required
        if column not in prices.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Thiếu các cột bắt buộc: {missing_columns}"
        )

    prices = (
        prices[required]
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if prices["Date"].isna().any():
        raise ValueError("Cột Date có ngày trống.")

    if prices["Date"].duplicated().any():
        raise ValueError("Cột Date có ngày bị trùng.")

    price_values = prices[TICKERS].apply(
        pd.to_numeric,
        errors="coerce"
    )

    if price_values.isna().any().any():
        bad = price_values.columns[
            price_values.isna().any()
        ].tolist()

        raise ValueError(
            f"Có giá thiếu hoặc không hợp lệ ở các mã: {bad}"
        )

    if not np.isfinite(
        price_values.to_numpy(dtype=np.float64)
    ).all():
        raise ValueError(
            "Giá có giá trị vô hạn hoặc không hữu hạn."
        )

    if (price_values <= 0).any().any():
        raise ValueError(
            "Giá cổ phiếu phải lớn hơn 0."
        )

    return price_values


# tính lợi suất hằng ngày, lợi suất kỳ vọng và ma trận hiệp phương sai
def calculate_parameters(
    price_values: pd.DataFrame
):

    daily_returns = price_values.pct_change(
        fill_method=None
    ).iloc[1:]

    if len(daily_returns) < 2:
        raise ValueError(
            "Cần ít nhất 3 phiên giá để tính hiệp phương sai."
        )

    # Expected Returns
    mu64 = (
        daily_returns.mean(axis=0)
        .to_numpy(dtype=np.float64)
        * TRADING_DAYS_PER_YEAR
    )

    # Covariance Matrix
    cov64 = (
        daily_returns.cov()
        .reindex(index=TICKERS, columns=TICKERS)
        .to_numpy(dtype=np.float64, copy=True)
    )

    cov64 *= TRADING_DAYS_PER_YEAR

    if not np.isfinite(mu64).all():
        raise ValueError("Mu có giá trị không hữu hạn.")

    if not np.isfinite(cov64).all():
        raise ValueError("Covariance có giá trị không hữu hạn.")

    # Convert to float32 for CUDA
    mu = mu64.astype(np.float32)
    cov = cov64.astype(np.float32)

    return daily_returns, mu, cov


# xuất dữ liệu thành file .txt
def save_parameters(
    output_dir: Path,
    mu: np.ndarray,
    cov: np.ndarray
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save mu.txt
    with (output_dir / "mu.txt").open(
        "w",
        encoding="utf-8",
        newline="\n"
    ) as file:

        file.write(f"{len(TICKERS)}\n")

        file.write(
            " ".join(
                f"{value:.9g}" for value in mu
            ) + "\n"
        )

    # Save cov.txt
    with (output_dir / "cov.txt").open(
        "w",
        encoding="utf-8",
        newline="\n"
    ) as file:

        for row in cov:

            file.write(
                " ".join(
                    f"{value:.9g}" for value in row
                ) + "\n"
            )


def main():

    root = Tk()
    root.withdraw()

    try:

        # Select input CSV
        input_file = select_input_file()

        if input_file is None:
            messagebox.showinfo(
                "Thông báo",
                "Bạn chưa chọn file dữ liệu."
            )
            return

        # Select output folder
        output_dir = select_output_folder()

        if output_dir is None:
            messagebox.showinfo(
                "Thông báo",
                "Bạn chưa chọn thư mục lưu kết quả."
            )
            return

        # Load and validate data
        price_values = load_prices(input_file)

        # Calculate financial parameters
        daily_returns, mu, cov = calculate_parameters(
            price_values
        )

        # Export results
        save_parameters(
            output_dir,
            mu,
            cov
        )

        result = (
            f"XỬ LÝ DỮ LIỆU THÀNH CÔNG!\n\n"
            f"File đầu vào: {input_file.name}\n"
            f"Số phiên giá: {len(price_values)}\n"
            f"Số quan sát lợi suất: {len(daily_returns)}\n"
            f"Số cổ phiếu: {len(TICKERS)}\n\n"
            f"Các mã: {', '.join(TICKERS)}\n\n"
            f"File kết quả:\n"
            f"{output_dir / 'mu.txt'}\n"
            f"{output_dir / 'cov.txt'}"
        )

        print(result)

        messagebox.showinfo(
            "Hoàn thành",
            result
        )

    except Exception as error:

        messagebox.showerror(
            "Lỗi xử lý dữ liệu",
            str(error)
        )

        print(f"LỖI: {error}")

    finally:
        root.destroy()


if __name__ == "__main__":
    main()