#ifndef CUDA_OPTIMIZED_CUH
#define CUDA_OPTIMIZED_CUH

#include "../common/portfolio_common.h"

// Chạy mô phỏng Monte Carlo M mẫu bằng CUDA Optimized
// Nạp mu và cov vào Shared Memory, Parallel Tree Reduction theo Block
OptimizationResult run_monte_carlo_optimized(
    const SimConfig& config,
    const std::vector<float>& h_mu,
    const std::vector<float>& h_cov,
    double& h2d_ms,
    double& kernel_ms,
    double& d2h_ms,
    std::vector<CandidateResult>* cloud_samples = nullptr,
    RunTimings* timings = nullptr
);

// Chạy kiểm thử số học tối ưu trên danh sách tỷ trọng cố định (A05 verification)
void run_verify_optimized(
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

#endif // CUDA_OPTIMIZED_CUH
