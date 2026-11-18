#include "cuda_optimized.cuh"
#include "portfolio_cuda_utils.cuh"

// Same RNG, metrics and tree reduction in both variants; only mu/cov storage differs.
template<int Capacity>
__global__ void kernel_mc_optimized(int K,long long M,float Rf,float R_min,unsigned long long seed,
    const float* __restrict__ mu,const float* __restrict__ cov,
    CandidateResult* best_sharpe,CandidateResult* best_risk,unsigned long long* valid_count,
    CandidateResult* cloud,int cloud_size) {
    extern __shared__ float shared[];
    const int tid=threadIdx.x;
    const int assets=Capacity==10?10:K;
    const long long idx=static_cast<long long>(blockIdx.x)*blockDim.x+tid;
    float* local_mu=shared;
    float* local_cov=local_mu+assets;
    for (int i=tid;i<assets;i+=blockDim.x) local_mu[i]=mu[i];
    for (int i=tid;i<assets*assets;i+=blockDim.x) local_cov[i]=cov[i];
    __syncthreads();
    float* reduction=local_cov+assets*assets;
    float w[Capacity];
    float rp=0,sigma=0,sharpe=INVALID_SHARPE;
    int valid=0;
    if (idx<M) {
        generate_dirichlet_weights<Capacity>(assets,seed,idx,w);
        evaluate_portfolio(assets,w,local_mu,local_cov,Rf,rp,sigma,sharpe);
        valid=rp>=R_min;
        if (idx<cloud_size) store_candidate(&cloud[idx],assets,w,rp,sigma,sharpe,valid);
    }
    // Inactive threads still participate in all block barriers.
    reduce_block(assets,M,idx,w,rp,sigma,sharpe,valid,reduction,best_sharpe,best_risk,valid_count);
}

__global__ void kernel_verify_optimized(int K,int N,float Rf,const float* weights,
    const float* mu,const float* cov,float* rp,float* sigma,float* sharpe) {
    extern __shared__ float shared[];
    float* local_mu=shared;
    float* local_cov=local_mu+K;
    for (int i=threadIdx.x;i<K;i+=blockDim.x) local_mu[i]=mu[i];
    for (int i=threadIdx.x;i<K*K;i+=blockDim.x) local_cov[i]=cov[i];
    __syncthreads();
    const int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (idx>=N) return;
    evaluate_portfolio(K,&weights[static_cast<size_t>(idx)*K],local_mu,local_cov,Rf,rp[idx],sigma[idx],sharpe[idx]);
}

OptimizationResult run_monte_carlo_optimized(const SimConfig& c,
    const std::vector<float>& mu,const std::vector<float>& cov,
    double& h2d,double& kernel,double& d2h,std::vector<CandidateResult>* cloud,RunTimings* timings) {
    return run_mc_gpu(c,mu,cov,h2d,kernel,d2h,cloud,timings,
        [&](int grid,const float* dm,const float* dc,CandidateResult* ds,CandidateResult* dr,
            unsigned long long* dv,CandidateResult* dp,int count) {
            const size_t bytes=reduction_shared_bytes(c.block_size)+static_cast<size_t>(c.K+c.K*c.K)*sizeof(float);
            if (c.K==10)
                kernel_mc_optimized<10><<<grid,c.block_size,bytes>>>(c.K,c.M,c.Rf,c.R_min,c.seed,dm,dc,ds,dr,dv,dp,count);
            else
                kernel_mc_optimized<MAX_ASSETS><<<grid,c.block_size,bytes>>>(c.K,c.M,c.Rf,c.R_min,c.seed,dm,dc,ds,dr,dv,dp,count);
        });
}

void run_verify_optimized(int K,int N,float Rf,const std::vector<float>& weights,
    const std::vector<float>& mu,const std::vector<float>& cov,
    std::vector<float>& rp,std::vector<float>& sigma,std::vector<float>& sharpe,double& kernel_ms) {
    run_verify_gpu(K,N,Rf,weights,mu,cov,rp,sigma,sharpe,kernel_ms,
        [&](int grid,int block,const float* dw,const float* dm,const float* dc,float* r,float* s,float* h) {
            kernel_verify_optimized<<<grid,block,static_cast<size_t>(K+K*K)*sizeof(float)>>>(K,N,Rf,dw,dm,dc,r,s,h);
        });
}
