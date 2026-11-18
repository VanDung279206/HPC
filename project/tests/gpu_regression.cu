#include "../src/cuda/cuda_naive.cuh"
#include "../src/cuda/cuda_optimized.cuh"
#include <iostream>

void require(bool condition,const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void check_candidate(const CandidateResult& p,int K,const std::vector<float>& mu,
                     const std::vector<float>& cov,float Rf) {
    std::vector<float> weights(p.w,p.w+K);
    validate_weights(1,K,weights);
    double r,s,h;
    evaluate_portfolio_reference(K,p.w,mu.data(),cov.data(),Rf,r,s,h);
    require(std::abs(r-p.rp)<VERIFY_TOLERANCE,"candidate return");
    require(std::abs(s-p.sigma)<VERIFY_TOLERANCE,"candidate risk");
    require(std::abs(h-p.sharpe)<VERIFY_TOLERANCE,"candidate Sharpe");
}
void check_case(SimConfig c,const std::vector<float>& mu,const std::vector<float>& cov) {
    c.cloud_sample_size=static_cast<int>(c.M)+200; // Includes the historical M < cloud_size regression.
    double h,k,d;RunTimings ta,tb;std::vector<CandidateResult> cloud;
    const auto a=run_monte_carlo_naive(c,mu,cov,h,k,d,&cloud,&ta);
    const auto b=run_monte_carlo_optimized(c,mu,cov,h,k,d,nullptr,&tb);
    require(cloud.size()==static_cast<size_t>(c.M),"cloud must be clamped to M");
    long long feasible=0;
    double best_sharpe=-std::numeric_limits<double>::infinity(),best_risk=std::numeric_limits<double>::infinity();
    for (const auto& p:cloud) {
        check_candidate(p,c.K,mu,cov,c.Rf);
        best_sharpe=std::max(best_sharpe,static_cast<double>(p.sharpe));
        if (p.valid_rmin) { ++feasible;best_risk=std::min(best_risk,static_cast<double>(p.sigma)); }
    }
    for (const auto* result:{&a,&b}) {
        require(result->valid_count==feasible,"feasible count differs from all generated samples");
        require(result->has_min_risk==(feasible>0),"missing/false feasible state");
        check_candidate(result->max_sharpe,c.K,mu,cov,c.Rf);
        require(std::abs(result->max_sharpe.sharpe-best_sharpe)<VERIFY_TOLERANCE,"max reduction");
        if (feasible) {
            check_candidate(result->min_risk,c.K,mu,cov,c.Rf);
            require(std::abs(result->min_risk.sigma-best_risk)<VERIFY_TOLERANCE,"min reduction");
        } else {
            require(result->min_risk.rp==0 && result->min_risk.sigma==0,"absent result must be initialized");
        }
    }
    require(ta.total_ms>=ta.kernel_ms && tb.total_ms>=tb.kernel_ms,"end-to-end timer excludes kernel");
}
int main(int argc,char** argv) {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) {
        std::cout<<"SKIP: no CUDA GPU\n";return 77;
    }
    try {
        const std::filesystem::path root=argc>1?argv[1]:".";
        int K=0,N=0,test_K=0;std::vector<float> mu,cov,weights;
        load_mu_cov((root/"data/processed/mu.txt").string(),(root/"data/processed/cov.txt").string(),K,mu,cov);
        load_test_weights((root/"tests/test_data/test_weights.txt").string(),N,test_K,weights);
        require(K==test_K,"fixed test K mismatch");
        for (int method=0;method<2;++method) {
            std::vector<float> r,s,h;double ms;
            if (method) run_verify_optimized(K,N,0.04f,weights,mu,cov,r,s,h,ms);
            else run_verify_naive(K,N,0.04f,weights,mu,cov,r,s,h,ms);
            for (int i=0;i<N;++i) {
                double rr,ss,hh;
                evaluate_portfolio_reference(K,&weights[static_cast<size_t>(i)*K],mu.data(),cov.data(),0.04f,rr,ss,hh);
                require(std::isfinite(r[i]) && std::abs(rr-r[i])<VERIFY_TOLERANCE,"fixed return");
                require(std::isfinite(s[i]) && std::abs(ss-s[i])<VERIFY_TOLERANCE,"fixed risk");
                require(std::isfinite(h[i]) && std::abs(hh-h[i])<VERIFY_TOLERANCE,"fixed Sharpe");
            }
        }
        std::cout<<"PASS: both verification kernels vs double reference, "<<N<<" fixed samples\n";
        int cases=0;
        for (int assets:{1,3,10,64}) {
            std::vector<float> simple_mu(assets,0.2f),simple_cov(assets*assets,0);
            for (int i=0;i<assets;++i) simple_cov[i*assets+i]=0.04f;
            for (int block:{64,128,256,512}) for (long long samples:{1LL,65LL,1003LL}) {
                SimConfig c;c.K=assets;c.block_size=block;c.M=samples;
                check_case(c,simple_mu,simple_cov);++cases;
            }
            SimConfig c;c.K=assets;c.M=65;c.R_min=1;check_case(c,simple_mu,simple_cov);++cases;
        }
        SimConfig real;real.K=K;real.M=10000;check_case(real,mu,cov);++cases;
        std::cout<<"PASS: "<<cases<<" reduction, partial-block, small-M and infeasible cases\n";
        for (int invalid:{0,8,192,1024}) {
            SimConfig c;c.block_size=invalid;double h,k,d;
            bool rejected=false;
            try { run_monte_carlo_naive(c,mu,cov,h,k,d); } catch(const std::runtime_error&) {rejected=true;}
            require(rejected,"invalid block must be rejected before launch");
            rejected=false;
            try { run_monte_carlo_optimized(c,mu,cov,h,k,d); } catch(const std::runtime_error&) {rejected=true;}
            require(rejected,"optimized invalid block must be rejected before launch");
        }
        SimConfig distribution;distribution.M=10000;distribution.cloud_sample_size=10000;
        std::vector<CandidateResult> cloud;double h,k,d;
        auto first=run_monte_carlo_optimized(distribution,mu,cov,h,k,d,&cloud);
        auto second=run_monte_carlo_optimized(distribution,mu,cov,h,k,d);
        for (int i=0;i<distribution.K;++i)
            require(first.max_sharpe.w[i]==second.max_sharpe.w[i],"same seed/config must reproduce the winner");
        double mean=0,second_moment=0;
        for (const auto& p:cloud) {mean+=p.w[0];second_moment+=p.w[0]*p.w[0];}
        mean/=cloud.size();second_moment/=cloud.size();
        require(std::abs(mean-0.1)<0.004,"Dirichlet mean");
        require(std::abs(second_moment-2.0/(10*11))<0.0015,"Dirichlet second moment");
        std::cout<<"PASS: launch validation, reproducibility and Dirichlet moments\n";
        return 0;
    } catch(const std::exception& error) { std::cerr<<"FAIL: "<<error.what()<<'\n';return 1; }
}
