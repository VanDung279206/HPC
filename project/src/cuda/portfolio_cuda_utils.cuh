#ifndef PORTFOLIO_CUDA_UTILS_CUH
#define PORTFOLIO_CUDA_UTILS_CUH

#include "../common/portfolio_common.h"
#include <curand_kernel.h>
#include <math_constants.h>

// Dynamic shared storage: two float arrays and three int arrays, one entry/thread.
constexpr size_t reduction_shared_bytes(int block) { return static_cast<size_t>(block)*20; }

template<int Capacity>
__device__ __forceinline__ void generate_dirichlet_weights(int K, unsigned long long seed,
                                                          long long sample, float* w) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed,static_cast<unsigned long long>(sample),0,&state);
    const int assets=Capacity==10?10:K;
    float sum=0;
    #pragma unroll
    for (int i=0;i<assets;++i) {
        // cuRAND returns U in (0,1]; -log(U) ~ Exp(1), normalized -> Dirichlet(1).
        w[i]=-logf(curand_uniform(&state));
        sum+=w[i];
    }
    #pragma unroll
    for (int i=0;i<assets;++i) w[i]=sum>0?w[i]/sum:1.0f/assets;
}

__device__ __forceinline__ void store_candidate(CandidateResult* out, int K, const float* w,
                                                float rp, float sigma, float sharpe, int valid) {
    out->rp=rp; out->sigma=sigma; out->sharpe=sharpe; out->valid_rmin=valid;
    for (int i=0;i<MAX_ASSETS;++i) out->w[i]=w && i<K?w[i]:0.0f;
}

__device__ __forceinline__ void reduce_block(int K, long long M, long long idx, const float* w,
                                            float rp, float sigma, float sharpe, int valid,
                                            float* storage, CandidateResult* best_sharpe,
                                            CandidateResult* best_risk, unsigned long long* valid_count) {
    const int tid=threadIdx.x, B=blockDim.x;
    float* scores=storage;
    float* risks=scores+B;
    int* score_ids=reinterpret_cast<int*>(risks+B);
    int* risk_ids=score_ids+B;
    int* counts=risk_ids+B;
    scores[tid]=idx<M?sharpe:-CUDART_INF_F;
    risks[tid]=idx<M && valid?sigma:CUDART_INF_F;
    score_ids[tid]=tid; risk_ids[tid]=tid; counts[tid]=idx<M?valid:0;
    __syncthreads();
    for (int offset=B/2;offset>0;offset/=2) {
        if (tid<offset) {
            if (scores[tid+offset]>scores[tid]) {
                scores[tid]=scores[tid+offset]; score_ids[tid]=score_ids[tid+offset];
            }
            if (risks[tid+offset]<risks[tid]) {
                risks[tid]=risks[tid+offset]; risk_ids[tid]=risk_ids[tid+offset];
            }
            counts[tid]+=counts[tid+offset];
        }
        __syncthreads();
    }
    if (tid==score_ids[0] && idx<M)
        store_candidate(&best_sharpe[blockIdx.x],K,w,rp,sigma,sharpe,valid);
    if (counts[0]>0) {
        if (tid==risk_ids[0] && idx<M)
            store_candidate(&best_risk[blockIdx.x],K,w,rp,sigma,sharpe,valid);
    } else if (tid==0) {
        store_candidate(&best_risk[blockIdx.x],K,nullptr,0,0,INVALID_SHARPE,0);
    }
    if (tid==0 && counts[0]>0) atomicAdd(valid_count,static_cast<unsigned long long>(counts[0]));
}

template<typename T> class DeviceBuffer {
public:
    T* data=nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count>std::numeric_limits<size_t>::max()/sizeof(T)) throw std::runtime_error("Device allocation overflow");
        if (count) CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&data),count*sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&)=delete;
    DeviceBuffer& operator=(const DeviceBuffer&)=delete;
    void reset() { if (data) { T* p=data; data=nullptr; CHECK_CUDA(cudaFree(p)); } }
};

class EventTimer {
    cudaEvent_t start_=nullptr,stop_=nullptr;
public:
    EventTimer() {
        CHECK_CUDA(cudaEventCreate(&start_));
        const auto error=cudaEventCreate(&stop_);
        if (error!=cudaSuccess) { cudaEventDestroy(start_); CHECK_CUDA(error); }
    }
    ~EventTimer() { if (start_) cudaEventDestroy(start_); if (stop_) cudaEventDestroy(stop_); }
    void start() { CHECK_CUDA(cudaEventRecord(start_)); }
    double stop() {
        CHECK_CUDA(cudaEventRecord(stop_));
        CHECK_CUDA(cudaEventSynchronize(stop_));
        float value=0; CHECK_CUDA(cudaEventElapsedTime(&value,start_,stop_));
        return value;
    }
};

inline int checked_grid(long long M,int block) {
    const long long grid=(M-1)/block+1;
    int device=0; cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDevice(&device)); CHECK_CUDA(cudaGetDeviceProperties(&prop,device));
    if (block>prop.maxThreadsPerBlock || grid>prop.maxGridSize[0])
        throw std::runtime_error("Launch exceeds this GPU's grid/block limits");
    return static_cast<int>(grid);
}

// Both implementations use exactly the same allocation, transfers, host reduction and timers.
template<typename Launcher>
OptimizationResult run_mc_gpu(const SimConfig& c, const std::vector<float>& mu,
                               const std::vector<float>& cov, double& h2d, double& kernel,
                               double& d2h, std::vector<CandidateResult>* cloud,
                               RunTimings* details, Launcher launch) {
    validate_config(c); validate_model(c.K,mu,cov);
    const int grid=checked_grid(c.M,c.block_size);
    const int cloud_size=cloud?static_cast<int>(std::min<long long>(c.cloud_sample_size,c.M)):0;
    const auto total_start=SteadyClock::now();
    RunTimings timing;
    OptimizationResult result{};
    {
        const auto allocation_start=SteadyClock::now();
        std::vector<CandidateResult> host_sharpe(grid),host_risk(grid);
        if (cloud) cloud->resize(cloud_size);
        DeviceBuffer<float> d_mu(c.K),d_cov(static_cast<size_t>(c.K)*c.K);
        DeviceBuffer<CandidateResult> d_sharpe(grid),d_risk(grid),d_cloud(cloud_size);
        DeviceBuffer<unsigned long long> d_valid(1);
        EventTimer timer;
        CHECK_CUDA(cudaMemset(d_valid.data,0,sizeof(unsigned long long)));
        CHECK_CUDA(cudaDeviceSynchronize());
        timing.allocation_ms=elapsed_ms(allocation_start);

        timer.start();
        CHECK_CUDA(cudaMemcpy(d_mu.data,mu.data(),mu.size()*sizeof(float),cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_cov.data,cov.data(),cov.size()*sizeof(float),cudaMemcpyHostToDevice));
        timing.h2d_ms=timer.stop();

        timer.start();
        launch(grid,d_mu.data,d_cov.data,d_sharpe.data,d_risk.data,d_valid.data,d_cloud.data,cloud_size);
        CHECK_CUDA(cudaGetLastError());
        timing.kernel_ms=timer.stop();

        unsigned long long valid=0;
        timer.start();
        CHECK_CUDA(cudaMemcpy(host_sharpe.data(),d_sharpe.data,grid*sizeof(CandidateResult),cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(host_risk.data(),d_risk.data,grid*sizeof(CandidateResult),cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&valid,d_valid.data,sizeof(valid),cudaMemcpyDeviceToHost));
        if (cloud_size)
            CHECK_CUDA(cudaMemcpy(cloud->data(),d_cloud.data,cloud_size*sizeof(CandidateResult),cudaMemcpyDeviceToHost));
        timing.d2h_ms=timer.stop();

        const auto reduce_start=SteadyClock::now();
        result.valid_count=static_cast<long long>(valid);
        result.max_sharpe.sharpe=-std::numeric_limits<float>::infinity();
        for (int b=0;b<grid;++b) {
            if (host_sharpe[b].sharpe>result.max_sharpe.sharpe) result.max_sharpe=host_sharpe[b];
            if (host_risk[b].valid_rmin && (!result.has_min_risk || host_risk[b].sigma<result.min_risk.sigma)) {
                result.min_risk=host_risk[b]; result.has_min_risk=true;
            }
        }
        if (!std::isfinite(result.max_sharpe.sharpe) ||
            result.has_min_risk!=(result.valid_count>0))
            throw std::runtime_error("Invalid reduction result from GPU");
        timing.host_reduce_ms=elapsed_ms(reduce_start);
        const auto cleanup_start=SteadyClock::now();
        d_mu.reset(); d_cov.reset(); d_sharpe.reset(); d_risk.reset(); d_cloud.reset(); d_valid.reset();
        timing.cleanup_ms=elapsed_ms(cleanup_start);
    } // Host vectors and timing events are also destroyed before the end-to-end clock stops.
    timing.total_ms=elapsed_ms(total_start);
    h2d=timing.h2d_ms; kernel=timing.kernel_ms; d2h=timing.d2h_ms;
    if (details) *details=timing;
    return result;
}

template<typename Launcher>
void run_verify_gpu(int K,int N,float Rf,const std::vector<float>& weights,
                    const std::vector<float>& mu,const std::vector<float>& cov,
                    std::vector<float>& rp,std::vector<float>& sigma,std::vector<float>& sharpe,
                    double& kernel_ms,Launcher launch) {
    validate_model(K,mu,cov); validate_weights(N,K,weights);
    if (!std::isfinite(Rf)) throw std::runtime_error("Rf must be finite");
    constexpr int block=256;
    const int grid=checked_grid(N,block);
    DeviceBuffer<float> dw(weights.size()),dm(K),dc(cov.size()),dr(N),ds(N),dh(N);
    CHECK_CUDA(cudaMemcpy(dw.data,weights.data(),weights.size()*sizeof(float),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dm.data,mu.data(),mu.size()*sizeof(float),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dc.data,cov.data(),cov.size()*sizeof(float),cudaMemcpyHostToDevice));
    EventTimer timer; timer.start();
    launch(grid,block,dw.data,dm.data,dc.data,dr.data,ds.data,dh.data);
    CHECK_CUDA(cudaGetLastError());
    kernel_ms=timer.stop();
    rp.resize(N); sigma.resize(N); sharpe.resize(N);
    CHECK_CUDA(cudaMemcpy(rp.data(),dr.data,N*sizeof(float),cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(sigma.data(),ds.data,N*sizeof(float),cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(sharpe.data(),dh.data,N*sizeof(float),cudaMemcpyDeviceToHost));
}
#endif
