/* scaled_throughput.cu */
/*
   This module mimics a deep learning workload by allocating vectors in GPU memory
   and performing matrix multiplication. The goal of this module is to charactrize 
   the GPU performance.

   NOTE: make sure to add the name of this file in the CMakeLists.txt.
   The run these commands to rebuild the modules.
   - rm -rf build
   - mkdir build
   - cd build
   - cmake .. -DNVBench_ENABLE_EXAMPLES=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
   - cmake --build . -j
*/

#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <cmath>



// CUDA matrix multiplication kernel O(N^3)
__global__ void matmul_kernel(const float *A, const float *B, float *C, int N)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N && col < N)
  {
    float sum = 0.0f;
    for (int k = 0; k < N; ++k)
      sum += A[row * N + k] * B[k * N + col];
    C[row * N + col] = sum;
  }
}

// Benchmark function
void matmul_datasize_bench(nvbench::state &state)
{
  // get the matrix size and data size in bytes
  const int N = static_cast<int>(state.get_int64("MatrixSize"));
  const std::size_t extra_bytes = static_cast<std::size_t>(state.get_int64("DataSizeBytes"));

  // get the GPU Id for debugging
  const int gpu_id = static_cast<int>(state.get_int64("GPU_ID"));
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (gpu_id >= device_count)
    throw std::runtime_error("Invalid GPU_ID axis value.");

  // NOTE: No cudaSetDevice call here; nvbench manages device context

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, gpu_id);

  printf("\nRunning on GPU %d: %s\n", gpu_id, prop.name);

  const std::size_t num_elements = static_cast<std::size_t>(N) * N;
  const std::size_t matrix_bytes = num_elements * sizeof(float);

  // allocate device memory using cudaMalloc directly
  float *A, *B, *C;
  cudaMalloc(&A, matrix_bytes);
  cudaMalloc(&B, matrix_bytes);
  cudaMalloc(&C, matrix_bytes);

  float *extra_d = nullptr;
  if (extra_bytes > 0)
  {
    cudaMalloc(&extra_d, extra_bytes);
  }

  // host buffers
  std::vector<float> hA(num_elements, 1.0f);
  std::vector<float> hB(num_elements, 1.0f);

  // specify the expected global memory writes and reads
  state.add_global_memory_writes<float>(num_elements * 2, "PCIe_Write");
  if (extra_bytes > 0)
  {
    state.add_global_memory_writes<float>(extra_bytes / sizeof(float), "Extra_Write");
  }

  // host to device copies
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
    cudaMemcpyAsync(A, hA.data(), matrix_bytes, cudaMemcpyHostToDevice, launch.get_stream());
    cudaMemcpyAsync(B, hB.data(), matrix_bytes, cudaMemcpyHostToDevice, launch.get_stream());
    if (extra_d && extra_bytes > 0)
    {
      std::vector<float> hExtra(extra_bytes / sizeof(float), 1.0f);
      cudaMemcpyAsync(extra_d, hExtra.data(), extra_bytes, cudaMemcpyHostToDevice, launch.get_stream());
    }
  });

  // kernel execution
  const dim3 block(16, 16);
  const dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
    matmul_kernel<<<grid, block, 0, launch.get_stream()>>>(A, B, C, N);
  });

  // device to host copies
  state.add_global_memory_reads<float>(num_elements, "PCIe_Read");
  if (extra_bytes > 0)
  {
    state.add_global_memory_reads<float>(extra_bytes / sizeof(float), "Extra_Read");
  }

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
    std::vector<float> hC(num_elements);
    cudaMemcpyAsync(hC.data(), C, matrix_bytes, cudaMemcpyDeviceToHost, launch.get_stream());
    if (extra_d && extra_bytes > 0)
    {
      std::vector<float> hExtra(extra_bytes / sizeof(float));
      cudaMemcpyAsync(hExtra.data(), extra_d, extra_bytes, cudaMemcpyDeviceToHost, launch.get_stream());
    }
  });

  // cleanup
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  if (extra_d)
    cudaFree(extra_d);

  // synchronize to ensure everything is complete
  cudaDeviceSynchronize();
}

NVBENCH_BENCH(matmul_datasize_bench)
  .add_int64_axis("GPU_ID", {0, 1})
  .add_int64_axis("MatrixSize", {16, 32})
  //.add_int64_axis("MatrixSize", {64, 128, 256, 512, 1024})
  .add_int64_axis("DataSizeBytes", {
    128L * 1024 * 1024,
    512L * 1024 * 1024,
    //1L * 1024 * 1024 * 1024,
    //2L * 1024 * 1024 * 1024
  });

