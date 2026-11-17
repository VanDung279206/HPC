#include "../common/portfolio_config.h"
#include "cuda_naive.cuh"
#include "cuda_optimized.cuh"
#include <array>
#include <iostream>

#ifndef PORTFOLIO_FAST_MATH
#define PORTFOLIO_FAST_MATH 0
#endif

namespace fs=std::filesystem;

void print_help() {
    std::cout<<"Portfolio Monte Carlo GPU (A04)\n"
        <<"  --project-root <project>  --config <json>  --output-dir <directory>\n"
        <<"  --mode all|verify|mc     --samples <M>    --block-size 64|128|256|512\n"
        <<"  --seed <uint64>          --rf <rate>      --r-min <return>\n"
        <<"  --cloud-size <N>         --warmup <N>     --repeats <N>\n"
        <<"  --pause (optional; default is noninteractive)\n"
        <<"Config values are loaded first; command-line values override them.\n";
}

void save_metadata(const CommandLine& o) {
    int device=0,driver=0,runtime=0;cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDevice(&device));CHECK_CUDA(cudaGetDeviceProperties(&prop,device));
    CHECK_CUDA(cudaDriverGetVersion(&driver));CHECK_CUDA(cudaRuntimeGetVersion(&runtime));
    auto out=open_csv(o.output/"run_metadata.txt");
    const auto& c=o.config;
    out<<"gpu="<<prop.name<<"\ncompute_capability="<<prop.major<<'.'<<prop.minor
       <<"\ndriver_api="<<driver<<"\nruntime_api="<<runtime
       <<"\nnvcc="<<__CUDACC_VER_MAJOR__<<'.'<<__CUDACC_VER_MINOR__<<'.'<<__CUDACC_VER_BUILD__
       <<"\nfast_math="<<PORTFOLIO_FAST_MATH<<"\nbuild="<<__DATE__<<' '<<__TIME__
       <<"\nproject_root="<<o.root.string()<<"\nconfig="<<o.config_path.string()
       <<"\nK="<<c.K<<"\nM="<<c.M<<"\nblock_size="<<c.block_size<<"\nseed="<<c.seed
       <<"\nRf="<<c.Rf<<"\nR_min="<<c.R_min<<"\ncloud_sample_size="<<c.cloud_sample_size
       <<"\nwarmup_runs="<<c.warmup_runs<<"\nrepeats="<<c.repeats
       <<"\nsampling=Dirichlet(1,...,1), normalized -log(U), Philox(seed,sample_id,0)"
       <<"\nbenchmark_cloud=disabled\nverification_absolute_tolerance="<<VERIFY_TOLERANCE<<'\n';
    out.flush();
}

bool run_verification(const CommandLine& o,const std::vector<float>& mu,
                      const std::vector<float>& cov,const std::vector<std::string>& tickers) {
    int N=0,test_K=0;std::vector<float> weights;
    load_test_weights((o.root/"tests/test_data/test_weights.txt").string(),N,test_K,weights);
    if (test_K!=o.config.K) throw std::runtime_error("test_weights K does not match mu/config K");
    std::array<std::vector<float>,3> naive,opt;
    double naive_ms=0,opt_ms=0;
    run_verify_naive(test_K,N,o.config.Rf,weights,mu,cov,naive[0],naive[1],naive[2],naive_ms);
    run_verify_optimized(test_K,N,o.config.Rf,weights,mu,cov,opt[0],opt[1],opt[2],opt_ms);
    auto out=open_csv(o.output/"tables/verify_gpu.csv");
    out<<"sample_id";
    for (const auto& t:tickers) out<<",w_"<<t;
    out<<",cpu_return,cpu_risk,cpu_sharpe,naive_return,naive_risk,naive_sharpe"
          ",optimized_return,optimized_risk,optimized_sharpe,naive_pass,optimized_pass\n";
    std::array<std::array<double,3>,2> errors{};
    std::array<int,2> failed{};
    for (int row=0;row<N;++row) {
        std::array<double,3> ref{};
        evaluate_portfolio_reference(test_K,&weights[static_cast<size_t>(row)*test_K],mu.data(),cov.data(),
                                     o.config.Rf,ref[0],ref[1],ref[2]);
        std::array<bool,2> pass{true,true};
        for (int method=0;method<2;++method) for (int metric=0;metric<3;++metric) {
            const double value=(method==0?naive:opt)[metric][row];
            const double difference=std::abs(ref[metric]-value);
            if (!std::isfinite(value) || !std::isfinite(ref[metric])) {
                errors[method][metric]=std::numeric_limits<double>::infinity();pass[method]=false;
            } else {
                errors[method][metric]=std::max(errors[method][metric],difference);
                if (difference>=VERIFY_TOLERANCE) pass[method]=false;
            }
        }
        out<<row;
        for (int col=0;col<test_K;++col) out<<','<<weights[static_cast<size_t>(row)*test_K+col];
        for (double value:ref) out<<','<<value;
        for (const auto& metric:naive) out<<','<<metric[row];
        for (const auto& metric:opt) out<<','<<metric[row];
        for (int method=0;method<2;++method) { out<<','<<(pass[method]?1:0);if(!pass[method])++failed[method]; }
        out<<'\n';
    }
    out.flush();
    for (int method=0;method<2;++method) {
        std::cout<<(method==0?"Naive":"Optimized")<<" verify: N="<<N
            <<" max_abs_error(return,risk,sharpe)="<<std::scientific
            <<errors[method][0]<<','<<errors[method][1]<<','<<errors[method][2]
            <<" failed="<<failed[method]<<'\n';
    }
    const bool pass=failed[0]==0 && failed[1]==0;
    std::cout<<(pass?"[PASS]":"[FAIL]")<<" Both kernels vs double CPU reference (absolute error < 1e-5)\n";
    return pass;
}

double median(std::vector<double> values) {
    std::sort(values.begin(),values.end());
    const size_t n=values.size();
    return n%2?values[n/2]:(values[n/2-1]+values[n/2])/2;
}
RunTimings median_timings(const std::vector<RunTimings>& timings) {
    RunTimings result;
    for (auto member:{&RunTimings::allocation_ms,&RunTimings::h2d_ms,&RunTimings::kernel_ms,
                     &RunTimings::d2h_ms,&RunTimings::host_reduce_ms,&RunTimings::cleanup_ms,&RunTimings::total_ms}) {
        std::vector<double> values;
        for (const auto& t:timings) values.push_back(t.*member);
        result.*member=median(values);
    }
    return result;
}
void benchmark_header(std::ostream& out) {
    out<<"method,K,M_simulations,block_size,seed,Rf,R_min,repeat,warmup_runs,allocation_ms,"
          "H2D_ms,kernel_ms,D2H_ms,host_reduce_ms,cleanup_ms,total_ms,kernel_speedup_vs_naive,total_speedup_vs_naive\n";
}
void benchmark_row(std::ostream& out,const char* method,const SimConfig& c,int repeat,
                   const RunTimings& t,double speed_kernel,double speed_total) {
    out<<method<<','<<c.K<<','<<c.M<<','<<c.block_size<<','<<c.seed<<','<<c.Rf<<','<<c.R_min<<','
       <<repeat<<','<<c.warmup_runs<<','<<t.allocation_ms<<','<<t.h2d_ms<<','<<t.kernel_ms<<','
       <<t.d2h_ms<<','<<t.host_reduce_ms<<','<<t.cleanup_ms<<','<<t.total_ms<<','
       <<speed_kernel<<','<<speed_total<<'\n';
}
double ratio(double baseline,double measured) { return measured>0?baseline/measured:0; }

void run_benchmark(const CommandLine& o,const std::vector<float>& mu,
                   const std::vector<float>& cov,const std::vector<std::string>& tickers) {
    const auto& c=o.config;
    double h=0,k=0,d=0;
    auto invoke=[&](bool optimized,RunTimings* timings=nullptr) {
        return optimized?run_monte_carlo_optimized(c,mu,cov,h,k,d,nullptr,timings):
                         run_monte_carlo_naive(c,mu,cov,h,k,d,nullptr,timings);
    };
    for (int i=0;i<c.warmup_runs;++i) { invoke(false);invoke(true); }
    std::array<std::vector<RunTimings>,2> trials;
    std::array<OptimizationResult,2> best;
    for (int repeat=0;repeat<c.repeats;++repeat) {
        // Alternate order to reduce systematic warm-up/clock ordering bias.
        for (int order=0;order<2;++order) {
            const int method=(order+repeat)%2;
            RunTimings timing;best[method]=invoke(method==1,&timing);trials[method].push_back(timing);
        }
        if (best[0].valid_count!=best[1].valid_count)
            throw std::runtime_error("Naive/optimized valid counts disagree for the same samples");
        // Shared and global variants must rank the same deterministic samples.
        for (int metric=0;metric<3;++metric) {
            const float a=metric==0?best[0].max_sharpe.rp:metric==1?best[0].max_sharpe.sigma:best[0].max_sharpe.sharpe;
            const float b=metric==0?best[1].max_sharpe.rp:metric==1?best[1].max_sharpe.sigma:best[1].max_sharpe.sharpe;
            if (!std::isfinite(a) || !std::isfinite(b) || std::abs(a-b)>=VERIFY_TOLERANCE)
                throw std::runtime_error("Naive/optimized max-Sharpe results disagree");
        }
    }
    auto raw=open_csv(o.output/"tables/benchmark_times.csv");
    benchmark_header(raw);
    for (int i=0;i<c.repeats;++i) for (int method=0;method<2;++method)
        benchmark_row(raw,method?"GPU_Optimized":"GPU_Naive",c,i+1,trials[method][i],
                      method?ratio(trials[0][i].kernel_ms,trials[1][i].kernel_ms):1,
                      method?ratio(trials[0][i].total_ms,trials[1][i].total_ms):1);
    raw.flush();
    std::array<RunTimings,2> med{median_timings(trials[0]),median_timings(trials[1])};
    auto summary=open_csv(o.output/"tables/benchmark_summary.csv");
    benchmark_header(summary);
    for (int method=0;method<2;++method) {
        benchmark_row(summary,method?"GPU_Optimized":"GPU_Naive",c,0,med[method],
                      method?ratio(med[0].kernel_ms,med[1].kernel_ms):1,
                      method?ratio(med[0].total_ms,med[1].total_ms):1);
        save_optimal_csv((o.output/"tables/optimal_portfolios.csv").string(),
                         method?"GPU_Optimized":"GPU_Naive",c.K,c.M,best[method].valid_count,med[method].total_ms,
                         best[method].max_sharpe,best[method].min_risk,method!=0,&tickers);
        std::cout<<std::fixed<<std::setprecision(6)<<(method?"Optimized":"Naive")
                 <<": median kernel="<<med[method].kernel_ms<<" ms, end-to-end="<<med[method].total_ms
                 <<" ms, max Sharpe="<<best[method].max_sharpe.sharpe<<'\n';
        if (best[method].has_min_risk)
            std::cout<<"  Min risk="<<best[method].min_risk.sigma<<" return="<<best[method].min_risk.rp
                     <<" feasible="<<best[method].valid_count<<'/'<<c.M<<'\n';
        else std::cout<<"  No feasible portfolio meets R_min; Min_Risk row is omitted.\n";
    }
    summary.flush();
    std::cout<<"Speedup optimized/naive: kernel="<<ratio(med[0].kernel_ms,med[1].kernel_ms)
             <<"x, end-to-end="<<ratio(med[0].total_ms,med[1].total_ms)<<"x\n";
    if (c.cloud_sample_size>0) {
        SimConfig cloud_config=c;
        cloud_config.M=std::min<long long>(c.M,c.cloud_sample_size);
        std::vector<CandidateResult> cloud;
        run_monte_carlo_optimized(cloud_config,mu,cov,h,k,d,&cloud);
        auto out=open_csv(o.output/"figures/cloud_points.csv");
        out<<"sample_id,return_Rp,risk_sigma,sharpe_S,valid_rmin";
        for (const auto& t:tickers) out<<",w_"<<t;
        out<<'\n';
        for (size_t i=0;i<cloud.size();++i) {
            const auto& p=cloud[i];out<<i<<','<<p.rp<<','<<p.sigma<<','<<p.sharpe<<','<<p.valid_rmin;
            for (int j=0;j<c.K;++j) out<<','<<p.w[j];
            out<<'\n';
        }
        out.flush();
        std::cout<<"Exported "<<cloud.size()<<" cloud points outside the benchmark.\n";
    } else {
        // Avoid leaving an old cloud from a previous run in a reused output directory.
        std::error_code error;
        fs::remove(o.output/"figures/cloud_points.csv",error);
        if (error) throw std::runtime_error("Cannot remove stale cloud output: "+error.message());
    }
}

int main(int argc,char** argv) {
    try {
        const auto options=parse_command_line(argc,argv);
        if (options.help) { print_help();return 0; }
        int K=0;std::vector<float> mu,cov;std::vector<std::string> tickers;
        load_mu_cov((options.root/"data/processed/mu.txt").string(),
                    (options.root/"data/processed/cov.txt").string(),K,mu,cov);
        if (K!=options.config.K) throw std::runtime_error("K in config differs from mu.txt");
        load_tickers((options.root/"data/processed/tickers.txt").string(),K,tickers);
        save_metadata(options);
        std::cout<<"Portfolio GPU A04: K="<<K<<" M="<<options.config.M
                 <<" block="<<options.config.block_size<<" seed="<<options.config.seed<<'\n';
        if ((options.mode=="verify" || options.mode=="all") && !run_verification(options,mu,cov,tickers))
            return 2;
        if (options.mode=="mc" || options.mode=="all") run_benchmark(options,mu,cov,tickers);
        std::cout<<"Completed. Output: "<<fs::absolute(options.output).string()<<'\n';
        if (options.pause) { std::cout<<"Press Enter to exit...\n";std::cin.get(); }
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<"[ERROR] "<<error.what()<<'\n';
        return 1;
    }
}
