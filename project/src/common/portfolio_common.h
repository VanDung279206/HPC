#ifndef PORTFOLIO_COMMON_H
#define PORTFOLIO_COMMON_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#if defined(__CUDACC__)
#define HD_INLINE __host__ __device__ inline
#else
#define HD_INLINE inline
#endif

inline void check_cuda(cudaError_t e, const char* expression, const char* file, int line) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(file)+":"+std::to_string(line)+" "+expression+": "+cudaGetErrorString(e));
}
#define CHECK_CUDA(call) check_cuda((call), #call, __FILE__, __LINE__)

constexpr int MAX_ASSETS = 64;
constexpr float INVALID_SHARPE = -1e9f;
constexpr double WEIGHT_TOLERANCE = 1e-6;
constexpr double VERIFY_TOLERANCE = 1e-5;

struct SimConfig {
    int K = 10;
    long long M = 1000000;
    float Rf = 0.04f;
    float R_min = 0.12f;
    unsigned long long seed = 42ULL;
    int block_size = 256;
    int cloud_sample_size = 2000;
    int warmup_runs = 1;
    int repeats = 5;
};
struct CandidateResult {
    float rp = 0.0f, sigma = 0.0f, sharpe = INVALID_SHARPE;
    int valid_rmin = 0;
    float w[MAX_ASSETS] = {};
};
struct OptimizationResult {
    CandidateResult max_sharpe{}, min_risk{};
    long long valid_count = 0;
    bool has_min_risk = false;
};
struct RunTimings {
    double allocation_ms=0, h2d_ms=0, kernel_ms=0, d2h_ms=0;
    double host_reduce_ms=0, cleanup_ms=0, total_ms=0;
};
using SteadyClock = std::chrono::steady_clock;
inline double elapsed_ms(SteadyClock::time_point start) {
    return std::chrono::duration<double,std::milli>(SteadyClock::now()-start).count();
}

inline long long parse_integer(const std::string& token, const std::string& name) {
    size_t used=0; long long value;
    try { value=std::stoll(token,&used); }
    catch (const std::exception&) { throw std::runtime_error("Invalid integer for "+name+": "+token); }
    if (used!=token.size()) throw std::runtime_error("Invalid integer for "+name+": "+token);
    return value;
}
inline int parse_int(const std::string& token, const std::string& name) {
    const auto value=parse_integer(token,name);
    if (value<std::numeric_limits<int>::min() || value>std::numeric_limits<int>::max())
        throw std::runtime_error(name+" is outside the supported integer range");
    return static_cast<int>(value);
}
inline unsigned long long parse_seed(const std::string& token) {
    if (token.empty() || token.front()=='-') throw std::runtime_error("seed must be nonnegative");
    size_t used=0; unsigned long long value;
    try { value=std::stoull(token,&used); }
    catch (const std::exception&) { throw std::runtime_error("Invalid seed: "+token); }
    if (used!=token.size()) throw std::runtime_error("Invalid seed: "+token);
    return value;
}
inline float parse_float(const std::string& token, const std::string& name) {
    size_t used=0; float value;
    try { value=std::stof(token,&used); }
    catch (const std::exception&) { throw std::runtime_error("Invalid number for "+name+": "+token); }
    if (used!=token.size() || !std::isfinite(value))
        throw std::runtime_error("Expected a finite number for "+name+": "+token);
    return value;
}
inline std::string read_token(std::istream& input, const std::string& context) {
    std::string token;
    if (!(input>>token)) throw std::runtime_error("Missing or unreadable "+context);
    return token;
}
inline void require_end(std::istream& input, const std::string& path) {
    std::string extra;
    if (input>>extra) throw std::runtime_error("Unexpected extra data in "+path+": "+extra);
    if (input.bad()) throw std::runtime_error("Read error in "+path);
}
inline void validate_config(const SimConfig& c) {
    if (c.K<1 || c.K>MAX_ASSETS) throw std::runtime_error("K must be in [1,64]");
    if (c.M<=0) throw std::runtime_error("M must be positive");
    if (c.block_size!=64 && c.block_size!=128 && c.block_size!=256 && c.block_size!=512)
        throw std::runtime_error("block_size must be 64, 128, 256 or 512");
    if ((c.M-1)/c.block_size+1>std::numeric_limits<int>::max())
        throw std::runtime_error("M exceeds the supported CUDA grid size");
    if (!std::isfinite(c.Rf) || !std::isfinite(c.R_min))
        throw std::runtime_error("Rf and R_min must be finite");
    if (c.cloud_sample_size<0) throw std::runtime_error("cloud_sample_size must be nonnegative");
    if (c.warmup_runs<0 || c.repeats<1)
        throw std::runtime_error("warmup_runs must be nonnegative and repeats must be positive");
}
inline void validate_model(int K, const std::vector<float>& mu, const std::vector<float>& cov) {
    if (K<1 || K>MAX_ASSETS || mu.size()!=static_cast<size_t>(K) || cov.size()!=static_cast<size_t>(K)*K)
        throw std::runtime_error("mu/cov dimensions do not match K");
    double scale=0;
    for (float x:mu) if (!std::isfinite(x)) throw std::runtime_error("mu contains a non-finite value");
    for (float x:cov) {
        if (!std::isfinite(x)) throw std::runtime_error("cov contains a non-finite value");
        scale=std::max(scale,std::abs(static_cast<double>(x)));
    }
    const double tolerance=std::max(scale,1e-12)*1e-6;
    std::vector<double> factor(static_cast<size_t>(K)*K,0);
    // Semidefinite Cholesky check, allowing float32 roundoff at zero pivots.
    for (int i=0;i<K;++i) for (int j=0;j<=i;++j) {
        if (std::abs(static_cast<double>(cov[i*K+j])-cov[j*K+i])>tolerance)
            throw std::runtime_error("cov must be symmetric");
        double value=0.5*(static_cast<double>(cov[i*K+j])+cov[j*K+i]);
        for (int p=0;p<j;++p) value-=factor[i*K+p]*factor[j*K+p];
        if (i==j) {
            if (value < -tolerance) throw std::runtime_error("cov must be positive semidefinite");
            factor[i*K+j]=value>tolerance ? std::sqrt(value):0;
        } else if (factor[j*K+j]>0) factor[i*K+j]=value/factor[j*K+j];
        else if (std::abs(value)>tolerance) throw std::runtime_error("cov must be positive semidefinite");
    }
}
inline bool load_mu_cov(const std::string& mu_path, const std::string& cov_path,
                        int& K, std::vector<float>& mu, std::vector<float>& cov) {
    std::ifstream fm(mu_path),fc(cov_path);
    if (!fm) throw std::runtime_error("Cannot open "+mu_path);
    if (!fc) throw std::runtime_error("Cannot open "+cov_path);
    K=parse_int(read_token(fm,"K in "+mu_path),"K");
    if (K<1 || K>MAX_ASSETS) throw std::runtime_error("K must be in [1,64]");
    mu.resize(K); cov.resize(static_cast<size_t>(K)*K);
    for (int i=0;i<K;++i) mu[i]=parse_float(read_token(fm,"mu["+std::to_string(i)+"] in "+mu_path),mu_path);
    for (int i=0;i<K*K;++i) cov[i]=parse_float(read_token(fc,"cov["+std::to_string(i)+"] in "+cov_path),cov_path);
    require_end(fm,mu_path); require_end(fc,cov_path);
    validate_model(K,mu,cov);
    return true;
}
inline bool load_tickers(const std::string& path, int K, std::vector<std::string>& tickers) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open "+path);
    tickers.clear(); std::string ticker;
    while (input>>ticker) {
        if (ticker.find_first_of(",\"")!=std::string::npos)
            throw std::runtime_error("Ticker contains an unsupported CSV character in "+path);
        if (std::find(tickers.begin(),tickers.end(),ticker)!=tickers.end())
            throw std::runtime_error("Duplicate ticker in "+path);
        tickers.push_back(ticker);
    }
    if (input.bad() || tickers.size()!=static_cast<size_t>(K))
        throw std::runtime_error("Expected exactly K tickers in "+path);
    return true;
}
inline void validate_weights(int N, int K, const std::vector<float>& weights) {
    if (N<1 || K<1 || K>MAX_ASSETS || weights.size()!=static_cast<size_t>(N)*K)
        throw std::runtime_error("Invalid test weight dimensions");
    for (int row=0;row<N;++row) {
        double sum=0;
        for (int col=0;col<K;++col) {
            float x=weights[static_cast<size_t>(row)*K+col];
            if (!std::isfinite(x) || x<0) throw std::runtime_error("Invalid weight in sample "+std::to_string(row));
            sum+=x;
        }
        if (std::abs(sum-1)>WEIGHT_TOLERANCE)
            throw std::runtime_error("Weights do not sum to 1 in sample "+std::to_string(row));
    }
}
inline bool load_test_weights(const std::string& path, int& N, int& K, std::vector<float>& weights) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open "+path);
    N=parse_int(read_token(input,"sample count in "+path),"sample count");
    K=parse_int(read_token(input,"K in "+path),"test K");
    if (N<1 || K<1 || K>MAX_ASSETS) throw std::runtime_error("Invalid test weight header in "+path);
    const size_t count=static_cast<size_t>(N)*K;
    if (count>std::filesystem::file_size(path))
        throw std::runtime_error("Test weight header exceeds the data available in "+path);
    weights.resize(count);
    for (size_t i=0;i<count;++i)
        weights[i]=parse_float(read_token(input,"weight "+std::to_string(i)+" in "+path),path);
    require_end(input,path); validate_weights(N,K,weights);
    return true;
}

HD_INLINE void evaluate_portfolio(int K, const float* w, const float* mu, const float* cov,
                                  float Rf, float& rp, float& sigma, float& sharpe) {
    rp=0; float variance=0;
    for (int i=0;i<K;++i) {
        rp+=w[i]*mu[i]; float row=0;
        for (int j=0;j<K;++j) row+=cov[i*K+j]*w[j];
        variance+=w[i]*row;
    }
#if defined(__CUDA_ARCH__)
    sigma=sqrtf(fmaxf(0.0f,variance));
#else
    sigma=std::sqrt(std::max(0.0f,variance));
#endif
    sharpe=sigma>1e-8f ? (rp-Rf)/sigma:INVALID_SHARPE;
}
inline void evaluate_portfolio_cpu(int K, const float* w, const float* mu, const float* cov,
                                   float Rf, float& rp, float& sigma, float& sharpe) {
    evaluate_portfolio(K,w,mu,cov,Rf,rp,sigma,sharpe);
}
inline void evaluate_portfolio_reference(int K, const float* w, const float* mu, const float* cov,
                                         float Rf, double& rp, double& sigma, double& sharpe) {
    rp=0; double variance=0;
    for (int i=0;i<K;++i) {
        rp+=static_cast<double>(w[i])*mu[i];
        for (int j=0;j<K;++j) variance+=static_cast<double>(w[i])*cov[i*K+j]*w[j];
    }
    sigma=std::sqrt(std::max(0.0,variance));
    sharpe=sigma>1e-8 ? (rp-static_cast<double>(Rf))/sigma:INVALID_SHARPE;
}
inline std::ofstream open_csv(const std::filesystem::path& path, bool append=false) {
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path,append?std::ios::app:std::ios::out);
    if (!output) throw std::runtime_error("Cannot write "+path.string());
    output.exceptions(std::ios::badbit|std::ios::failbit);
    output<<std::setprecision(std::numeric_limits<double>::max_digits10);
    return output;
}
inline void save_optimal_csv(const std::string& path, const std::string& method,
                             int K, long long M, long long valid_count, double elapsed,
                             const CandidateResult& best_sharpe, const CandidateResult& best_min_risk,
                             bool append=true, const std::vector<std::string>* tickers=nullptr) {
    const bool exists=std::filesystem::exists(path);
    auto out=open_csv(path,append);
    if (!exists || !append) {
        out<<"method,criterion,M_simulations,valid_rmin_count,time_ms,return_Rp,risk_sigma,sharpe_S";
        for (int i=0;i<K;++i) out<<",w_"<<(tickers?tickers->at(i):std::to_string(i+1));
        out<<'\n';
    }
    auto row=[&](const char* criterion,const CandidateResult& r) {
        out<<method<<','<<criterion<<','<<M<<','<<valid_count<<','<<elapsed<<','<<r.rp<<','<<r.sigma<<','<<r.sharpe;
        for (int i=0;i<K;++i) out<<','<<r.w[i];
        out<<'\n';
    };
    row("Max_Sharpe",best_sharpe);
    if (valid_count>0) row("Min_Risk_Rmin",best_min_risk);
    out.flush();
}
#endif
