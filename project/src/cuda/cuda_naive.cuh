#ifndef CUDA_NAIVE_CUH
#define CUDA_NAIVE_CUH

#include "../common/portfolio_common.h"

// Chạy mô phỏng Monte Carlo M mẫu bằng CUDA Naive
// Đọc mu và cov từ Global Memory
OptimizationResult run_monte_carlo_naive(
    const SimConfig& config,
    const std::vector<float>& h_mu,
    const std::vector<float>& h_cov,
    double& h2d_ms,
    double& kernel_ms,
    double& d2h_ms,
    std::vector<CandidateResult>* cloud_samples = nullptr,
    RunTimings* timings = nullptr
);

// Chạy kiểm thử số học trên danh sách tỷ trọng cố định (A05 verification)
void run_verify_naive(
    int K,
    int num_samples,
    float Rf,
    const std::vector<float>& h_weights,
    const std::vector<float>& h_mu,
    const std::vector<float>& h_cov,
    std::vector<float>& h_rp,
    std::vector<float>& h_sigma,
    std::vector<float>& h_sharpe,
    double& kernel_ms
);

#endif // CUDA_NAIVE_CUH
