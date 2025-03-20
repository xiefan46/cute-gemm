#include <cuda.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <cute/tensor.hpp>

template <typename T>
void gen_rand_data(T* data, int n);


int main() {
  srand(10086);

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


  TiledMMA mma = make_tiled_mma(mma_atom{}, make_layout(Shape<_2, _2, _1>{}), make_layout(Shape<_1, _2, _1>{}));

//  std::cout<<" size mma: "<<size(mma)<<std::endl;
//  print(mma);

  print_latex(mma);

}

template <typename T>
void gen_rand_data(T* data, int n) {
  for (int i = 0; i < n; i++) {
    float v = (rand() % 200 - 100) * 0.01;
    data[i] = static_cast<T>(v);
  }
}