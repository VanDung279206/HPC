/**
 * cpu_mc.cpp
 * Monte Carlo Portfolio Optimization - CPU Implementation
 *
 * Task A04: Static Arithmetic Testing + Sequential & OpenMP versions
 *
 * Features:
 *  - Static test mode: reads 10,000 fixed weights from test_weights.txt
 *    and writes verify_cpu.csv for cross-validation with GPU (Task A05)
 *  - Sequential mode: single-thread std::mt19937(seed=42), M scenarios
 *  - OpenMP mode: parallel with per-thread independent RNG
 *  - Outputs: verify_cpu.csv, cloud_points.csv, optimal_portfolios.csv
 *
 * Compile (MSVC):
 *   cl /O2 /openmp /std:c++17 cpu_mc.cpp /Fe:cpu_mc.exe
 *
 * Compile (GCC/MinGW):
 *   g++ -O2 -fopenmp -std=c++17 cpu_mc.cpp -o cpu_mc
 *
 * Run:
 *   cpu_mc.exe <project_root>
 *   e.g.:  cpu_mc.exe E:\HPC\HPC\project
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;

// ─────────────────────────── Constants ────────────────────────────
static constexpr int    K            = 10;        // number of assets
static constexpr int    M_SCENARIOS  = 1000000;   // scenarios for sequential / OpenMP
static constexpr int    N_TEST       = 10000;     // samples in static test
static constexpr int    N_CLOUD      = 3000;      // cloud points to export
static constexpr double RF           = 0.04;      // risk-free rate
static constexpr double R_MIN        = 0.12;      // minimum acceptable return
static constexpr int    SEED         = 42;
static constexpr double EPS_WEIGHT   = 1e-6;      // |sum(w)-1| tolerance

// ─────────────────────────── Data Structures ──────────────────────
struct Portfolio {
    double ret   = 0.0;   // E[Rp]
    double risk  = 0.0;   // sigma_p
    double sharpe= 0.0;   // (Rp - Rf) / sigma_p
    std::vector<double> weights;
};

// ─────────────────────────── File Loaders ─────────────────────────
int load_K(const std::string& mu_path) {
    std::ifstream f(mu_path);
    if (!f) throw std::runtime_error("Cannot open " + mu_path);
    int k; f >> k;
    return k;
}

std::vector<double> load_mu(const std::string& mu_path, int K) {
    std::ifstream f(mu_path);
    if (!f) throw std::runtime_error("Cannot open " + mu_path);
    int k; f >> k;
    std::vector<double> mu(K);
    for (int i = 0; i < K; ++i) f >> mu[i];
    return mu;
}

// cov.txt: K×K matrix, row-major, space-separated
std::vector<double> load_cov(const std::string& cov_path, int K) {
    std::ifstream f(cov_path);
    if (!f) throw std::runtime_error("Cannot open " + cov_path);
    std::vector<double> cov(K * K);
    for (int i = 0; i < K * K; ++i) f >> cov[i];
    return cov;
}

std::vector<std::string> load_tickers(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path);
    std::vector<std::string> tickers;
    std::string s;
    while (f >> s) tickers.push_back(s);
    return tickers;
}

// test_weights.txt: first line = "N K", then N rows of K weights
std::vector<std::vector<double>> load_test_weights(const std::string& path, int K) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path);
    int n, k; f >> n >> k;
    if (k != K) throw std::runtime_error("test_weights K mismatch");
    std::vector<std::vector<double>> ws(n, std::vector<double>(K));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < K; ++j)
            f >> ws[i][j];
    return ws;
}

// ─────────────────────────── Metric Computation ───────────────────
/**
 * Given weights w[K], mu[K], cov[K*K]:
 *   ret    = sum_i w[i]*mu[i]
 *   var    = sum_i sum_j w[i]*cov[i*K+j]*w[j]
 *   risk   = sqrt(var)
 *   sharpe = (ret - RF) / risk
 */
inline void compute_metrics(const double* w,
                             const double* mu,
                             const double* cov,
                             int K,
                             double& ret,
                             double& risk,
                             double& sharpe)
{
    ret = 0.0;
    for (int i = 0; i < K; ++i) ret += w[i] * mu[i];

    double var = 0.0;
    for (int i = 0; i < K; ++i) {
        double wCov = 0.0;
        for (int j = 0; j < K; ++j)
            wCov += cov[i * K + j] * w[j];
        var += w[i] * wCov;
    }
    risk   = std::sqrt(std::max(var, 0.0));
    sharpe = (risk > 1e-12) ? (ret - RF) / risk : -std::numeric_limits<double>::max();
}

// ─────────────────────────── Weight Generator ─────────────────────
/**
 * Generates one valid weight vector via Dirichlet-like sampling:
 *   draw K exponentials, normalize to sum=1 (guarantees w_i>=0, sum=1).
 * Uses mt19937 rng passed by reference.
 */
inline void generate_weights(std::mt19937& rng,
                              std::uniform_real_distribution<double>& dist,
                              double* w,
                              int K)
{
    double s = 0.0;
    for (int i = 0; i < K; ++i) {
        // -ln(U) ~ Exp(1)
        w[i] = -std::log(dist(rng));
        s   += w[i];
    }
    for (int i = 0; i < K; ++i) w[i] /= s;
}

// ─────────────────────────── Static Test Mode ─────────────────────
void run_static_test(const std::string& test_weights_path,
                     const std::string& out_csv_path,
                     const std::vector<double>& mu,
                     const std::vector<double>& cov,
                     const std::vector<std::string>& tickers,
                     int K)
{
    std::cout << "\n=== STATIC ARITHMETIC TEST (verify_cpu.csv) ===\n";

    auto ws = load_test_weights(test_weights_path, K);
    int N   = static_cast<int>(ws.size());
    std::cout << "  Loaded " << N << " fixed weight sets\n";

    std::ofstream csv(out_csv_path);
    if (!csv) throw std::runtime_error("Cannot write " + out_csv_path);

    // Header
    csv << "sample_id";
    for (int i = 0; i < K; ++i) csv << ",w_" << tickers[i];
    csv << ",return,risk,sharpe\n";
    csv << std::fixed << std::setprecision(8);

    int bad = 0;
    for (int s = 0; s < N; ++s) {
        const double* w = ws[s].data();

        // Validate sum
        double wsum = 0.0;
        for (int i = 0; i < K; ++i) wsum += w[i];
        if (std::fabs(wsum - 1.0) > EPS_WEIGHT) {
            std::cerr << "  [WARN] sample " << s << " sum=" << wsum << " (skipped)\n";
            ++bad;
            continue;
        }
        // Validate no-short-selling
        bool valid = true;
        for (int i = 0; i < K; ++i)
            if (w[i] < 0.0) { valid = false; break; }
        if (!valid) {
            std::cerr << "  [WARN] sample " << s << " has negative weight (skipped)\n";
            ++bad;
            continue;
        }

        double ret, risk, sharpe;
        compute_metrics(w, mu.data(), cov.data(), K, ret, risk, sharpe);

        csv << s;
        for (int i = 0; i < K; ++i) csv << "," << w[i];
        csv << "," << ret << "," << risk << "," << sharpe << "\n";
    }

    std::cout << "  Valid samples written : " << (N - bad) << "\n";
    std::cout << "  Skipped (bad weights) : " << bad        << "\n";
    std::cout << "  Output -> " << out_csv_path << "\n";
}

// ─────────────────────────── Sequential MC ────────────────────────
struct MCResult {
    Portfolio max_sharpe;
    Portfolio min_risk;
    std::vector<Portfolio> cloud;   // first N_CLOUD points
    double time_ms = 0.0;
};

MCResult run_cpu_sequential(const std::vector<double>& mu,
                             const std::vector<double>& cov,
                             int K,
                             int M)
{
    std::cout << "\n=== SEQUENTIAL CPU Monte Carlo ===\n";
    std::cout << "  M = " << M << " scenarios, seed = " << SEED << "\n";

    MCResult result;
    result.max_sharpe.sharpe = -std::numeric_limits<double>::max();
    result.min_risk.risk     =  std::numeric_limits<double>::max();
    result.min_risk.ret      = -1.0;

    result.cloud.reserve(N_CLOUD);

    std::mt19937 rng(SEED);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::vector<double> w(K);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int m = 0; m < M; ++m) {
        generate_weights(rng, dist, w.data(), K);

        double ret, risk, sharpe;
        compute_metrics(w.data(), mu.data(), cov.data(), K, ret, risk, sharpe);

        // Max Sharpe
        if (sharpe > result.max_sharpe.sharpe) {
            result.max_sharpe.sharpe  = sharpe;
            result.max_sharpe.ret     = ret;
            result.max_sharpe.risk    = risk;
            result.max_sharpe.weights = std::vector<double>(w.begin(), w.end());
        }
        // Min Risk with Rp >= R_MIN
        if (ret >= R_MIN && risk < result.min_risk.risk) {
            result.min_risk.risk     = risk;
            result.min_risk.ret      = ret;
            result.min_risk.sharpe   = sharpe;
            result.min_risk.weights  = std::vector<double>(w.begin(), w.end());
        }
        // Cloud sample
        if (m < N_CLOUD) {
            Portfolio p;
            p.ret = ret; p.risk = risk; p.sharpe = sharpe;
            p.weights = std::vector<double>(w.begin(), w.end());
            result.cloud.push_back(std::move(p));
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    result.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "  Time  : " << std::fixed << std::setprecision(2)
              << result.time_ms << " ms\n";
    std::cout << "  Max Sharpe : S=" << result.max_sharpe.sharpe
              << " Rp=" << result.max_sharpe.ret
              << " sigma=" << result.max_sharpe.risk << "\n";
    if (result.min_risk.ret >= 0.0)
        std::cout << "  Min Risk   : sigma=" << result.min_risk.risk
                  << " Rp=" << result.min_risk.ret
                  << " S=" << result.min_risk.sharpe << "\n";
    else
        std::cout << "  Min Risk   : No valid portfolio found (Rp >= " << R_MIN << ")\n";

    return result;
}

// ─────────────────────────── OpenMP MC ────────────────────────────
MCResult run_cpu_openmp(const std::vector<double>& mu,
                        const std::vector<double>& cov,
                        int K,
                        int M)
{
#ifndef _OPENMP
    std::cerr << "[WARN] OpenMP not available – falling back to sequential\n";
    return run_cpu_sequential(mu, cov, K, M);
#else
    int nthreads = omp_get_max_threads();
    std::cout << "\n=== OPENMP CPU Monte Carlo ===\n";
    std::cout << "  M = " << M << " scenarios, threads = " << nthreads << "\n";

    // Per-thread local bests
    struct LocalBest {
        Portfolio max_sharpe;
        Portfolio min_risk;
        std::vector<Portfolio> cloud;
    };

    std::vector<LocalBest> locals(nthreads);
    for (auto& lb : locals) {
        lb.max_sharpe.sharpe = -std::numeric_limits<double>::max();
        lb.min_risk.risk     =  std::numeric_limits<double>::max();
        lb.min_risk.ret      = -1.0;
    }

    // We want the first N_CLOUD globally-ordered samples saved to cloud.
    // We use a shared vector protected by a critical section only for first N_CLOUD.
    std::vector<Portfolio> global_cloud;
    global_cloud.reserve(N_CLOUD);

    auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        int tid  = omp_get_thread_num();
        // Each thread gets its own RNG seeded with SEED + tid
        std::mt19937 rng(SEED + tid);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        std::vector<double> w(K);

        LocalBest& lb = locals[tid];

        #pragma omp for schedule(static)
        for (int m = 0; m < M; ++m) {
            generate_weights(rng, dist, w.data(), K);

            double ret, risk, sharpe;
            compute_metrics(w.data(), mu.data(), cov.data(), K, ret, risk, sharpe);

            if (sharpe > lb.max_sharpe.sharpe) {
                lb.max_sharpe = {ret, risk, sharpe,
                                 std::vector<double>(w.begin(), w.end())};
            }
            if (ret >= R_MIN && risk < lb.min_risk.risk) {
                lb.min_risk = {ret, risk, sharpe,
                               std::vector<double>(w.begin(), w.end())};
            }

            // Collect global cloud (first N_CLOUD across all threads, iteration order)
            if (m < N_CLOUD) {
                #pragma omp critical
                {
                    if (static_cast<int>(global_cloud.size()) < N_CLOUD) {
                        Portfolio p{ret, risk, sharpe,
                                    std::vector<double>(w.begin(), w.end())};
                        global_cloud.push_back(std::move(p));
                    }
                }
            }
        }
    } // end parallel

    auto t1 = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Merge local bests
    MCResult result;
    result.time_ms = time_ms;
    result.max_sharpe.sharpe = -std::numeric_limits<double>::max();
    result.min_risk.risk     =  std::numeric_limits<double>::max();
    result.min_risk.ret      = -1.0;

    for (auto& lb : locals) {
        if (lb.max_sharpe.sharpe > result.max_sharpe.sharpe)
            result.max_sharpe = lb.max_sharpe;
        if (lb.min_risk.ret >= 0.0 && lb.min_risk.risk < result.min_risk.risk)
            result.min_risk = lb.min_risk;
    }
    result.cloud = std::move(global_cloud);

    std::cout << "  Time  : " << std::fixed << std::setprecision(2)
              << result.time_ms << " ms\n";
    std::cout << "  Max Sharpe : S=" << result.max_sharpe.sharpe
              << " Rp=" << result.max_sharpe.ret
              << " sigma=" << result.max_sharpe.risk << "\n";
    if (result.min_risk.ret >= 0.0)
        std::cout << "  Min Risk   : sigma=" << result.min_risk.risk
                  << " Rp=" << result.min_risk.ret
                  << " S=" << result.min_risk.sharpe << "\n";
    else
        std::cout << "  Min Risk   : No valid portfolio found (Rp >= " << R_MIN << ")\n";

    return result;
#endif
}

// ─────────────────────────── CSV Writers ──────────────────────────
void write_cloud_csv(const std::string& path,
                     const std::vector<Portfolio>& cloud,
                     const std::vector<std::string>& tickers,
                     int K)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write " + path);
    f << "sample_id";
    for (int i = 0; i < K; ++i) f << ",w_" << tickers[i];
    f << ",return,risk,sharpe\n";
    f << std::fixed << std::setprecision(8);
    for (int i = 0; i < static_cast<int>(cloud.size()); ++i) {
        const auto& p = cloud[i];
        f << i;
        for (int j = 0; j < K; ++j) f << "," << p.weights[j];
        f << "," << p.ret << "," << p.risk << "," << p.sharpe << "\n";
    }
    std::cout << "  Cloud points (" << cloud.size() << ") -> " << path << "\n";
}

void write_optimal_csv(const std::string& path,
                       const MCResult& seq_res,
                       const MCResult& omp_res,
                       const std::vector<std::string>& tickers,
                       int K,
                       double speedup)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write " + path);
    f << std::fixed << std::setprecision(8);

    // Header
    f << "version,criterion";
    for (int i = 0; i < K; ++i) f << ",w_" << tickers[i];
    f << ",return,risk,sharpe,time_ms,speedup\n";

    auto write_row = [&](const std::string& ver,
                         const std::string& crit,
                         const Portfolio& p,
                         double t_ms,
                         double sp)
    {
        f << ver << "," << crit;
        for (int i = 0; i < K; ++i) f << "," << p.weights[i];
        f << "," << p.ret << "," << p.risk << "," << p.sharpe
          << "," << t_ms << "," << sp << "\n";
    };

    write_row("sequential", "max_sharpe", seq_res.max_sharpe, seq_res.time_ms, 1.0);
    if (seq_res.min_risk.ret >= 0.0)
        write_row("sequential", "min_risk",   seq_res.min_risk,   seq_res.time_ms, 1.0);

    write_row("openmp",     "max_sharpe", omp_res.max_sharpe, omp_res.time_ms, speedup);
    if (omp_res.min_risk.ret >= 0.0)
        write_row("openmp",     "min_risk",   omp_res.min_risk,   omp_res.time_ms, speedup);

    std::cout << "  Optimal portfolios -> " << path << "\n";
}

// ─────────────────────────── Main ─────────────────────────────────
int main(int argc, char* argv[])
{
    std::string root = (argc > 1) ? argv[1] : ".";
    // Normalize path separators
    for (auto& c : root) if (c == '\\') c = '/';
    // Strip trailing slash (tránh double-slash khi nối đường dẫn)
    while (root.size() > 1 && root.back() == '/') root.pop_back();

    std::string data_dir    = root + "/data/processed";
    std::string test_dir    = root + "/tests/test_data";
    std::string results_dir = root + "/results/tables";

    // Ensure results dir exists
    fs::create_directories(results_dir);

    // ── Load input data ──
    std::string mu_path   = data_dir  + "/mu.txt";
    std::string cov_path  = data_dir  + "/cov.txt";
    std::string tick_path = data_dir  + "/tickers.txt";
    std::string tw_path   = test_dir  + "/test_weights.txt";

    int K_actual = load_K(mu_path);
    if (K_actual != K) {
        std::cerr << "[ERROR] K in mu.txt (" << K_actual
                  << ") != compiled K (" << K << ")\n";
        return 1;
    }

    auto mu      = load_mu(mu_path, K);
    auto cov     = load_cov(cov_path, K);
    auto tickers = load_tickers(tick_path);

    std::cout << "Portfolio MC Optimizer (CPU)\n";
    std::cout << "  Assets : ";
    for (auto& t : tickers) std::cout << t << " ";
    std::cout << "\n  K=" << K << "  Rf=" << RF << "  R_min=" << R_MIN << "\n";

    // ── 1. Static Test Mode ──
    run_static_test(tw_path,
                    results_dir + "/verify_cpu.csv",
                    mu, cov, tickers, K);

    // ── 2a. Sequential ──
    MCResult seq_res = run_cpu_sequential(mu, cov, K, M_SCENARIOS);

    // ── 2b. OpenMP ──
    MCResult omp_res = run_cpu_openmp(mu, cov, K, M_SCENARIOS);

    double speedup = (omp_res.time_ms > 0.0)
                   ? seq_res.time_ms / omp_res.time_ms
                   : 0.0;
    std::cout << "\n  Speedup_OpenMP = " << std::fixed << std::setprecision(2)
              << speedup << "x\n";

    // ── Write cloud (from sequential, first N_CLOUD) ──
    write_cloud_csv(results_dir + "/cloud_points.csv",
                    seq_res.cloud, tickers, K);

    // ── Write optimal portfolios ──
    write_optimal_csv(results_dir + "/optimal_portfolios.csv",
                      seq_res, omp_res, tickers, K, speedup);

    std::cout << "\nAll outputs written to " << results_dir << "\n";
    return 0;
}
