#include <cuda.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <cute/tensor.hpp>

template <typename T>
void gen_rand_data(T* data, int n);


int main() {
  srand(10086);

  cudaDeviceProp props;
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (props.major < 8) {
    std::cout << "This example requires an Ampere GPU or newer (CC >= 80)" << std::endl;
    // Return 0 so tests pass if run on unsupported architectures or CUDA Toolkits.
    return 0;
  }

  const int m = 81920;
  const int n = 256;
  const int k = 256;

  using T = cute::half_t;
  using namespace cute;

//  T* Aptr_d;
//  T* Bptr_d;
//  T* Cptr_d;
//
//  cudaMalloc(&Aptr_d, sizeof(T) * m * k);
//  cudaMalloc(&Bptr_d, sizeof(T) * n * k);
//  cudaMalloc(&Cptr_d, sizeof(T) * m * n);
//
//  T* Aptr_h;
//  T* Bptr_h;
//  gen_rand_data(Aptr_h, m * k);
//  gen_rand_data(Bptr_h, n * k);
//
//  cudaMemcpy(Aptr_d, Aptr_h, sizeof(T) * m * k, cudaMemcpyHostToDevice);
//  cudaMemcpy(Bptr_d, Bptr_h, sizeof(T) * n * k, cudaMemcpyHostToDevice);

  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  using MMA = decltype(make_tiled_mma(mma_atom{}, make_layout(Shape<_2, _2, _1>{}), make_layout(Shape<_1, _2, _1>{})));

  std::cout<<" size mma: "<<size(MMA{})<<std::endl;
}

template <typename T>
void gen_rand_data(T* data, int n) {
  for (int i = 0; i < n; i++) {
    float v = (rand() % 200 - 100) * 0.01;
    data[i] = static_cast<T>(v);
  }
}