#include "CCubicDomain.cuh"           //ds domain structure
#include <iostream>                   //ds cout
#include <cuda_runtime.h>             //ds needed for eclipse indexer
#include <device_launch_parameters.h> //ds needed for eclipse indexer

#define N 20


// GPU kernels:
__global__ void kernel1(int* a)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    a[idx] = 7;
}

__global__ void kernel2(int* b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    b[idx] = blockIdx.x;
}

__global__ void kernel3(int* c)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    c[idx] = threadIdx.x;
}


int main()
{
    NBody::CVector vecTest;

    int h_a[N], h_b[N],h_c[N];  // h stands for host (stuffs on CPU)
    int* d_pa, *d_pb, *d_pc;    // d stands for device (stuffs on GPU)

    //allocate the memory on the GPU
    cudaMalloc( (void**)&d_pa, N*sizeof(int) );
    cudaMalloc( (void**)&d_pb, N*sizeof(int) );
    cudaMalloc( (void**)&d_pc, N*sizeof(int) );

    cudaMemset(d_pa, 0, N);
    cudaMemset(d_pb, 0, N);
    cudaMemset(d_pc, 0, N);

    // call the GPU kernels
    kernel1<<<0,0>>>(d_pa);
    kernel2<<<0,0>>>(d_pb);
    kernel3<<<0,0>>>(d_pc);

    // copy the arrays back from the GPU to the CPU
    cudaMemcpy(h_a, d_pa, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_pb, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c, d_pc, N*sizeof(int), cudaMemcpyDeviceToHost);

    // display the results
    std::cout << " Results from kernel1:" << std::endl;
    for (int i = 0; i<N; i++)
        std::cout<< h_a[i] << " ";
    std::cout<< std::endl;

    std::cout << " Results from kernel2:" << std::endl;
    for (int i = 0; i<N; i++)
        std::cout<< h_b[i] << " ";
    std::cout<< std::endl;

    std::cout << " Results from kernel3:" << std::endl;
    for (int i = 0; i<N; i++)
        std::cout<< h_c[i] << " ";
    std::cout << std::endl;


    //free the memory allocated on the GPU
    cudaFree(d_pa);
    cudaFree(d_pb);
    cudaFree(d_pc);

    return 0;

}
