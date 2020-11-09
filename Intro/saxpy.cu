#include <stdio.h>

//device code : kernel code: execute by multiple threads in parallel
// n,a,i variables will be stored by each thread in a register
// pointers x and y: must be pointer to the device memory address space.
__global__
void saxpy(int n, float a, float *x, float *y) //saxpy is the kernel that runs in parallel on the GPU
{
  // predefined variables [blockDim: contains dimensions of each thread block for kernel launch] [threadIdx and blockIdx: contain the index of the thread within its thread block and the thread block within the grid]
  // and the built-in device variables blockDim, blockIdx, and threadIdx used to identify and differentiate GPU threads that execute the kernel in parallel.
  int i = blockIdx.x*blockDim.x + threadIdx.x; // generated a global index that is used to access elements of the arrays
  if (i < n) y[i] = a*x[i] + y[i]; //performs the element-wise work of the SAXPY, and other than the bounds check, it is identical to the inner loop of a host implementation of SAXPY 
}

int main(void) //main function is the host code
{
  int N = 1<<20;
  float *x, *y, *d_x, *d_y; //passed d_x and d_y to the kernel when we launched to the device in host code
  
  //declares two pair of arrays
  x = (float*)malloc(N*sizeof(float)); //pointer x: points to the host array, allocated with malloc 
  y = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float)); //d_x array point to the device array allocated with the cudaMalloc function from CUDA runtime API
  cudaMalloc(&d_y, N*sizeof(float));

  //the host code then initializes the host array
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f; // setting x to an array of ones
    y[i] = 2.0f; // setting y to an array of twos
  }
  
  // to initialize the device arrays, we simply copy the data from x and y to the corresponding device arrayd d_x and d_y using cudaMemcpy
  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);  // use cudaMemcpyHostToDevice to specify that the first (destination) argument is a device pointer and the second (source) argument is a host pointer
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  // informaiton between triple chevrons is the execution configuration, which dictates how many device threads execute the kernel in parallel
  // The first argument in the execution configuration specifies the number of thread blocks in the grid, and the second specifies the number of threads in a thread block.
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y); // saxpy kernel is launched by this statement

  //after running the kernel, to get the results back to the host, we copy from the device array pointed to d_y to the host array pointed by y using cudaMemcpy with cudaMemcpyDeviceToHost
  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost); 

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);


  //after we finish, we should free any allocated memory.
  cudaFree(d_x); //for device memory allocated with cudaMalloc(), simply call cudaFree()
  cudaFree(d_y);
  free(x); //for host memory, use free() as usual
  free(y);
}

