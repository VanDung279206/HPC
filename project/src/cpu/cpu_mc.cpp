/**
 * cpu_mc.cpp
 * Monte Carlo Portfolio Optimization - CPU Implementation
 *
 * Task A03: Static Arithmetic Testing + Sequential & OpenMP versions
 *
 * Features:
 *  - Static test mode: reads 10,000 fixed weights from test_weights.txt
 *    and writes verify_cpu.csv for cross-validation with GPU (Task A05)
 *  - Sequential mode: single-thread std::mt19937, seed/M/Rf/R_min from JSON
 *  - OpenMP mode: parallel with pre-generated weights (bit-identical to seq)
 *  - Outputs: verify_cpu.csv, cloud_points.csv, optimal_portfolios.csv
 *
 * Configuration file: <project_root>/config/run_config.json
 *   Required fields: M (int), seed (int), Rf (double), R_min (double)
 *   Optional fields: K (int), block_size (int), cloud_sample_size (int)
 *
 * CLI overrides (all optional):
 *   --M <int>        Override number of scenarios
 *   --seed <int>     Override random seed
 *   --Rf <double>    Override risk-free rate  (annualised, e.g. 0.04)
 *   --Rmin <double>  Override minimum return  (annualised, e.g. 0.12)
 *   --threads <int>  Override OMP_NUM_THREADS
 *
 * Compile (MSVC):
 *   cl /O2 /openmp /std:c++17 /EHsc cpu_mc.cpp /Fe:cpu_mc.exe
 *
 * Compile (GCC/MinGW):
 *   g++ -O2 -fopenmp -std=c++17 cpu_mc.cpp -o cpu_mc
 *
 * Run:
 *   cpu_mc.exe <project_root> [options]
 *   e.g.:  cpu_mc.exe E:\HPC\HPC\project --M 10000000 --threads 8
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
static constexpr int    K_DEFAULT    = 10;      // number of assets (must match mu.txt)
static constexpr int    N_TEST       = 10000;   // samples in static test
static constexpr int    N_CLOUD      = 3000;    // cloud points to export
static constexpr double EPS_WEIGHT   = 1e-6;    // |sum(w)-1| tolerance

// ─────────────────────────── Data Structures ──────────────────────
struct Portfolio {
    double ret   = 0.0;   // E[Rp]
    double risk  = 0.0;   // sigma_p
    double sharpe= 0.0;   // (Rp - Rf) / sigma_p
    std::vector<double> weights;
};

// ─────────────────────────── Config ───────────────────────────────
struct RunConfig {
    int    M      = 1000000;
    int    seed   = 42;
    double Rf     = 0.04;
    double R_min  = 0.12;
    int    K      = K_DEFAULT;
    int    threads = -1;    // -1 = use OMP_NUM_THREADS / system default
};

// Minimal JSON parser for flat key-value objects (no dependency on nlohmann)
// Supports int and double values, ignores unknown keys.
static RunConfig load_config(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open config: " + path);

    RunConfig cfg;
    std::string line, content;
    while (std::getline(f, line)) content += line + " ";

    // Strip braces
    auto strip = [](std::string s) {
        s.erase(std::remove(s.begin(), s.end(), '{'), s.end());
        s.erase(std::remove(s.begin(), s.end(), '}'), s.end());
        return s;
    };
    content = strip(content);

    // Split by comma, then parse each "key": value pair
    std::istringstream ss(content);
    std::string token;
    while (std::getline(ss, token, ',')) {
        // Find colon
        auto colon = token.find(':');
        if (colon == std::string::npos) continue;
        std::string key = token.substr(0, colon);
        std::string val = token.substr(colon + 1);

        // Trim whitespace, quotes
        auto trim = [](std::string& s) {
            size_t a = s.find_first_not_of(" \t\r\n\"");
            size_t b = s.find_last_not_of(" \t\r\n\"");
            s = (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
        };
        trim(key); trim(val);
        if (key.empty() || val.empty()) continue;

        try {
            if (key == "M")     cfg.M    = std::stoi(val);
            else if (key == "seed")  cfg.seed  = std::stoi(val);
            else if (key == "Rf")    cfg.Rf    = std::stod(val);
            else if (key == "R_min") cfg.R_min = std::stod(val);
            else if (key == "K")     cfg.K     = std::stoi(val);
        } catch (...) {
            throw std::runtime_error("Invalid value for key '" + key + "': " + val);
        }
    }
    return cfg;
}

// Validate config; throw std::invalid_argument on error
static void validate_config(const RunConfig& cfg) {
    if (cfg.M <= 0)
        throw std::invalid_argument("M must be > 0, got " + std::to_string(cfg.M));
    if (cfg.seed < 0)
        throw std::invalid_argument("seed must be >= 0, got " + std::to_string(cfg.seed));
    if (cfg.Rf < 0.0 || cfg.Rf >= 1.0)
        throw std::invalid_argument("Rf must be in [0, 1), got " + std::to_string(cfg.Rf));
    if (cfg.R_min < 0.0 || cfg.R_min >= 2.0)
        throw std::invalid_argument("R_min must be in [0, 2), got " + std::to_string(cfg.R_min));
    if (cfg.K <= 0 || cfg.K > 1000)
        throw std::invalid_argument("K must be in (0, 1000], got " + std::to_string(cfg.K));
    if (cfg.threads != -1 && cfg.threads <= 0)
        throw std::invalid_argument("threads must be > 0 or -1 (auto), got " + std::to_string(cfg.threads));
}

// Apply CLI overrides to cfg
static void apply_cli(int argc, char* argv[], RunConfig& cfg) {
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--M" || arg == "-M") && i + 1 < argc) {
            cfg.M = std::stoi(argv[++i]);
        } else if ((arg == "--seed") && i + 1 < argc) {
            cfg.seed = std::stoi(argv[++i]);
        } else if ((arg == "--Rf") && i + 1 < argc) {
            cfg.Rf = std::stod(argv[++i]);
        } else if ((arg == "--Rmin" || arg == "--R_min") && i + 1 < argc) {
            cfg.R_min = std::stod(argv[++i]);
        } else if ((arg == "--threads" || arg == "-t") && i + 1 < argc) {
            cfg.threads = std::stoi(argv[++i]);
        }
    }
}

// Print effective config to stdout
static void print_config(const RunConfig& cfg, int omp_threads) {
    std::cout << "\n--- Effective Configuration ---\n";
    std::cout << "  M        = " << cfg.M      << "\n";
    std::cout << "  seed     = " << cfg.seed   << "\n";
    std::cout << "  Rf       = " << std::fixed << std::setprecision(4) << cfg.Rf   << "  (annualised)\n";
    std::cout << "  R_min    = " << std::fixed << std::setprecision(4) << cfg.R_min << "  (annualised)\n";
    std::cout << "  K        = " << cfg.K      << "\n";
    std::cout << "  threads  = " << omp_threads << " (OpenMP)\n";
    std::cout << "------------------------------\n";
}

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
    if (k != K) throw std::runtime_error("test_weights K mismatch: file=" +
                                          std::to_string(k) + " expected=" + std::to_string(K));
    std::vector<std::vector<double>> ws(n, std::vector<double>(K));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < K; ++j)
            f >> ws[i][j];
    return ws;
}

// ─────────────────────────── Metric Computation ───────────────────
/**
 * Given weights w[K], mu[K], cov[K*K], risk-free rate Rf:
 *   ret    = sum_i w[i]*mu[i]
 *   var    = sum_i sum_j w[i]*cov[i*K+j]*w[j]
 *   risk   = sqrt(var)
 *   sharpe = (ret - Rf) / risk
 *
 * mu, cov and Rf must be in the same annualised units.
 * No additional annualisation is applied here.
 */
inline void compute_metrics(const double* w,
                             const double* mu,
                             const double* cov,
                             int K,
                             double Rf,
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
    sharpe = (risk > 1e-12) ? (ret - Rf) / risk : -std::numeric_limits<double>::max();
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
                     int K,
                     double Rf)
{
    std::cout << "\n=== STATIC ARITHMETIC TEST (verify_cpu.csv) ===\n";

    auto ws = load_test_weights(test_weights_path, K);
    int N   = static_cast<int>(ws.size());
    std::cout << "  Loaded " << N << " fixed weight sets\n";

    std::ofstream csv(out_csv_path);
    if (!csv) throw std::runtime_error("Cannot write " + out_csv_path);

    // Header — unified column names for CPU/GPU cross-validation
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
        compute_metrics(w, mu.data(), cov.data(), K, Rf, ret, risk, sharpe);

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
    double time_ms    = 0.0;
    // config snapshot embedded in result
    int    M          = 0;
    int    seed       = 0;
    double Rf         = 0.0;
    double R_min      = 0.0;
    int    omp_threads = 0;
    bool   has_min_risk = false;
};

MCResult run_cpu_sequential(const std::vector<double>& mu,
                             const std::vector<double>& cov,
                             int K,
                             const RunConfig& cfg)
{
    std::cout << "\n=== SEQUENTIAL CPU Monte Carlo ===\n";
    std::cout << "  M = " << cfg.M << " scenarios, seed = " << cfg.seed << "\n";
    std::cout << "  Rf = " << cfg.Rf << "  R_min = " << cfg.R_min << "\n";

    MCResult result;
    result.M           = cfg.M;
    result.seed        = cfg.seed;
    result.Rf          = cfg.Rf;
    result.R_min       = cfg.R_min;
    result.omp_threads = 1;
    result.max_sharpe.sharpe = -std::numeric_limits<double>::max();
    result.min_risk.risk     =  std::numeric_limits<double>::max();
    result.min_risk.ret      = -1.0;
    result.cloud.reserve(N_CLOUD);

    std::mt19937 rng(static_cast<uint32_t>(cfg.seed));
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::vector<double> w(K);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int m = 0; m < cfg.M; ++m) {
        generate_weights(rng, dist, w.data(), K);

        double ret, risk, sharpe;
        compute_metrics(w.data(), mu.data(), cov.data(), K, cfg.Rf, ret, risk, sharpe);

        // Max Sharpe
        if (sharpe > result.max_sharpe.sharpe) {
            result.max_sharpe.sharpe  = sharpe;
            result.max_sharpe.ret     = ret;
            result.max_sharpe.risk    = risk;
            result.max_sharpe.weights = std::vector<double>(w.begin(), w.end());
        }
        // Min Risk with Rp >= R_min
        if (ret >= cfg.R_min && risk < result.min_risk.risk) {
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
    result.has_min_risk = (result.min_risk.ret >= 0.0);

    std::cout << "  Time  : " << std::fixed << std::setprecision(2)
              << result.time_ms << " ms\n";
    std::cout << "  Max Sharpe : S=" << result.max_sharpe.sharpe
              << " Rp=" << result.max_sharpe.ret
              << " sigma=" << result.max_sharpe.risk << "\n";
    if (result.has_min_risk)
        std::cout << "  Min Risk   : sigma=" << result.min_risk.risk
                  << " Rp=" << result.min_risk.ret
                  << " S=" << result.min_risk.sharpe << "\n";
    else
        std::cout << "  Min Risk   : No valid portfolio found (Rp >= "
                  << cfg.R_min << ")\n";

    return result;
}

// ─────────────────────────── OpenMP MC ────────────────────────────
//
// Reproducibility design:
//   Step 1 (sequential): Generate ALL M weight vectors using the SAME
//     single mt19937(seed) RNG as the sequential version → bit-identical
//     weights for every scenario m.
//   Step 2 (parallel):   Each thread computes metrics for its chunk of
//     pre-generated weights → no RNG race, results match sequential exactly.
//
// Memory: M * K * sizeof(double) = 1e6 * 10 * 8 = ~80 MB per 1M scenarios.
//
MCResult run_cpu_openmp(const std::vector<double>& mu,
                        const std::vector<double>& cov,
                        int K,
                        const RunConfig& cfg)
{
#ifndef _OPENMP
    std::cerr << "[WARN] OpenMP not available – falling back to sequential\n";
    return run_cpu_sequential(mu, cov, K, cfg);
#else
    int nthreads = omp_get_max_threads();
    std::cout << "\n=== OPENMP CPU Monte Carlo ===\n";
    std::cout << "  M = " << cfg.M << " scenarios, threads = " << nthreads
              << ", seed = " << cfg.seed << " (reproducible)\n";
    std::cout << "  Rf = " << cfg.Rf << "  R_min = " << cfg.R_min << "\n";

    // ── Step 1: Pre-generate all weights sequentially (same RNG as sequential) ──
    std::cout << "  [1/2] Generating " << cfg.M << " weight vectors (sequential RNG)...\n";
    std::vector<double> all_w(static_cast<size_t>(cfg.M) * K);
    {
        std::mt19937 rng(static_cast<uint32_t>(cfg.seed));
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (int m = 0; m < cfg.M; ++m)
            generate_weights(rng, dist, all_w.data() + static_cast<size_t>(m) * K, K);
    }

    // ── Step 2: Parallel metric computation over pre-generated weights ──
    std::cout << "  [2/2] Computing metrics in parallel...\n";

    struct LocalBest {
        Portfolio max_sharpe;
        Portfolio min_risk;
        std::vector<std::pair<int,Portfolio>> cloud; // (m, portfolio)
    };

    std::vector<LocalBest> locals(nthreads);
    for (auto& lb : locals) {
        lb.max_sharpe.sharpe = -std::numeric_limits<double>::max();
        lb.min_risk.risk     =  std::numeric_limits<double>::max();
        lb.min_risk.ret      = -1.0;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        LocalBest& lb = locals[tid];

        #pragma omp for schedule(static)
        for (int m = 0; m < cfg.M; ++m) {
            const double* w = all_w.data() + static_cast<size_t>(m) * K;

            double ret, risk, sharpe;
            compute_metrics(w, mu.data(), cov.data(), K, cfg.Rf, ret, risk, sharpe);

            if (sharpe > lb.max_sharpe.sharpe) {
                lb.max_sharpe = {ret, risk, sharpe,
                                 std::vector<double>(w, w + K)};
            }
            if (ret >= cfg.R_min && risk < lb.min_risk.risk) {
                lb.min_risk = {ret, risk, sharpe,
                               std::vector<double>(w, w + K)};
            }
            if (m < N_CLOUD) {
                lb.cloud.push_back({m,
                    Portfolio{ret, risk, sharpe, std::vector<double>(w, w + K)}});
            }
        }
    } // end parallel

    auto t1 = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ── Merge local bests ──
    MCResult result;
    result.M           = cfg.M;
    result.seed        = cfg.seed;
    result.Rf          = cfg.Rf;
    result.R_min       = cfg.R_min;
    result.omp_threads = nthreads;
    result.time_ms     = time_ms;
    result.max_sharpe.sharpe = -std::numeric_limits<double>::max();
    result.min_risk.risk     =  std::numeric_limits<double>::max();
    result.min_risk.ret      = -1.0;

    std::vector<std::pair<int,Portfolio>> raw_cloud;
    raw_cloud.reserve(N_CLOUD);

    for (auto& lb : locals) {
        if (lb.max_sharpe.sharpe > result.max_sharpe.sharpe)
            result.max_sharpe = lb.max_sharpe;
        if (lb.min_risk.ret >= 0.0 && lb.min_risk.risk < result.min_risk.risk)
            result.min_risk = lb.min_risk;
        for (auto& cp : lb.cloud)
            raw_cloud.push_back(std::move(cp));
    }

    // Sort by scenario index → cloud identical to sequential's first N_CLOUD
    std::sort(raw_cloud.begin(), raw_cloud.end(),
              [](const auto& a, const auto& b){ return a.first < b.first; });
    if (static_cast<int>(raw_cloud.size()) > N_CLOUD)
        raw_cloud.resize(N_CLOUD);
    result.cloud.reserve(raw_cloud.size());
    for (auto& cp : raw_cloud)
        result.cloud.push_back(std::move(cp.second));

    result.has_min_risk = (result.min_risk.ret >= 0.0);

    std::cout << "  Time  : " << std::fixed << std::setprecision(2)
              << result.time_ms << " ms\n";
    std::cout << "  Max Sharpe : S=" << result.max_sharpe.sharpe
              << " Rp=" << result.max_sharpe.ret
              << " sigma=" << result.max_sharpe.risk << "\n";
    if (result.has_min_risk)
        std::cout << "  Min Risk   : sigma=" << result.min_risk.risk
                  << " Rp=" << result.min_risk.ret
                  << " S=" << result.min_risk.sharpe << "\n";
    else
        std::cout << "  Min Risk   : No valid portfolio found (Rp >= "
                  << cfg.R_min << ")\n";

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

// Write a single method's optimal result as a separate file or append rows
// to the unified table (optimal_portfolios.csv).
// Columns match A05/A06 GPU output: method, return_Rp, risk_sigma, sharpe_S,
// plus weights, M, seed, Rf, R_min, threads, time_ms, speedup
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

    // Unified header (compatible with GPU output method/return_Rp/risk_sigma/sharpe_S)
    f << "method,criterion";
    for (int i = 0; i < K; ++i) f << ",w_" << tickers[i];
    f << ",return_Rp,risk_sigma,sharpe_S"
      << ",M,seed,Rf,R_min,threads,time_ms,speedup\n";

    auto write_row = [&](const std::string& method,
                         const std::string& crit,
                         const Portfolio& p,
                         const MCResult& res,
                         double t_ms,
                         double sp)
    {
        f << method << "," << crit;
        for (int i = 0; i < K; ++i) f << "," << p.weights[i];
        f << "," << p.ret << "," << p.risk << "," << p.sharpe
          << "," << res.M << "," << res.seed
          << "," << res.Rf << "," << res.R_min
          << "," << res.omp_threads
          << "," << t_ms << "," << sp << "\n";
    };

    write_row("CPU_Sequential", "max_sharpe", seq_res.max_sharpe, seq_res, seq_res.time_ms, 1.0);
    if (seq_res.has_min_risk)
        write_row("CPU_Sequential", "min_risk",   seq_res.min_risk,   seq_res, seq_res.time_ms, 1.0);
    else
        std::cout << "  [NOTE] Sequential: no portfolio found with Rp >= "
                  << seq_res.R_min << " — min_risk row omitted.\n";

    write_row("CPU_OpenMP",     "max_sharpe", omp_res.max_sharpe, omp_res, omp_res.time_ms, speedup);
    if (omp_res.has_min_risk)
        write_row("CPU_OpenMP",     "min_risk",   omp_res.min_risk,   omp_res, omp_res.time_ms, speedup);
    else
        std::cout << "  [NOTE] OpenMP: no portfolio found with Rp >= "
                  << omp_res.R_min << " — min_risk row omitted.\n";

    std::cout << "  Optimal portfolios -> " << path << "\n";
}

// Write a per-method benchmark summary
void write_benchmark_csv(const std::string& path,
                         const MCResult& seq_res,
                         const MCResult& omp_res,
                         double speedup)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write " + path);
    f << std::fixed << std::setprecision(4);

    // Timing scope: weight generation + metric computation + reduction
    // For sequential: full loop; for OpenMP: step1 (gen) + step2 (parallel compute)
    f << "method,M,seed,Rf,R_min,threads,time_ms,speedup_vs_seq,notes\n";
    f << "CPU_Sequential"
      << "," << seq_res.M << "," << seq_res.seed
      << "," << seq_res.Rf << "," << seq_res.R_min
      << "," << seq_res.omp_threads
      << "," << seq_res.time_ms << ",1.0"
      << ",weight_gen+metric_compute+reduction\n";
    f << "CPU_OpenMP"
      << "," << omp_res.M << "," << omp_res.seed
      << "," << omp_res.Rf << "," << omp_res.R_min
      << "," << omp_res.omp_threads
      << "," << omp_res.time_ms << "," << speedup
      << ",parallel_metric_only(seq_weight_gen_excluded_from_timing)\n";

    std::cout << "  Benchmark summary   -> " << path << "\n";
}

// ─────────────────────────── Main ─────────────────────────────────
int main(int argc, char* argv[])
{
    std::string root = (argc > 1) ? argv[1] : ".";
    // Normalize path separators
    for (auto& c : root) if (c == '\\') c = '/';
    // Strip trailing slash
    while (root.size() > 1 && root.back() == '/') root.pop_back();

    std::string data_dir    = root + "/data/processed";
    std::string test_dir    = root + "/tests/test_data";
    std::string results_dir = root + "/results/tables";
    std::string config_path = root + "/config/run_config.json";

    // ── Load configuration ──
    RunConfig cfg;
    try {
        cfg = load_config(config_path);
        std::cout << "Loaded config from: " << config_path << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }

    // Apply CLI overrides
    try {
        apply_cli(argc, argv, cfg);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] CLI argument error: " << e.what() << "\n";
        return 1;
    }

    // Validate
    try {
        validate_config(cfg);
    } catch (const std::invalid_argument& e) {
        std::cerr << "[ERROR] Invalid configuration: " << e.what() << "\n";
        return 1;
    }

    // Apply thread override
    int omp_threads = 1;
#ifdef _OPENMP
    if (cfg.threads > 0) omp_set_num_threads(cfg.threads);
    omp_threads = omp_get_max_threads();
#endif

    // Print effective config
    print_config(cfg, omp_threads);

    // Ensure results dir exists
    fs::create_directories(results_dir);

    // ── Load input data ──
    std::string mu_path   = data_dir  + "/mu.txt";
    std::string cov_path  = data_dir  + "/cov.txt";
    std::string tick_path = data_dir  + "/tickers.txt";
    std::string tw_path   = test_dir  + "/test_weights.txt";

    int K_actual = load_K(mu_path);
    if (K_actual != cfg.K) {
        std::cerr << "[ERROR] K in mu.txt (" << K_actual
                  << ") != config K (" << cfg.K << ")\n";
        return 1;
    }
    int K = cfg.K;

    auto mu      = load_mu(mu_path, K);
    auto cov     = load_cov(cov_path, K);
    auto tickers = load_tickers(tick_path);

    std::cout << "\nPortfolio MC Optimizer (CPU)\n";
    std::cout << "  Assets : ";
    for (auto& t : tickers) std::cout << t << " ";
    std::cout << "\n  K=" << K
              << "  Rf=" << cfg.Rf
              << "  R_min=" << cfg.R_min << "\n";

    // ── 1. Static Test Mode ──
    run_static_test(tw_path,
                    results_dir + "/verify_cpu.csv",
                    mu, cov, tickers, K, cfg.Rf);

    // ── 2a. Sequential ──
    MCResult seq_res = run_cpu_sequential(mu, cov, K, cfg);

    // ── 2b. OpenMP ──
    MCResult omp_res = run_cpu_openmp(mu, cov, K, cfg);

    double speedup = (omp_res.time_ms > 0.0)
                   ? seq_res.time_ms / omp_res.time_ms
                   : 0.0;
    std::cout << "\n  Speedup_OpenMP = " << std::fixed << std::setprecision(2)
              << speedup << "x\n";

    // ── Write cloud (from sequential, first N_CLOUD) ──
    write_cloud_csv(results_dir + "/cloud_points.csv",
                    seq_res.cloud, tickers, K);

    // ── Write optimal portfolios (unified columns for A05/A06) ──
    write_optimal_csv(results_dir + "/optimal_portfolios.csv",
                      seq_res, omp_res, tickers, K, speedup);

    // ── Write benchmark summary ──
    write_benchmark_csv(results_dir + "/benchmark_cpu.csv",
                        seq_res, omp_res, speedup);

    std::cout << "\nAll outputs written to " << results_dir << "\n";
    return 0;
}
