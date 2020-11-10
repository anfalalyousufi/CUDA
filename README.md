# CUDA

# What is CUDA?
It is the C/C++ interface to the CUDA parallel computing platform. It is an extension of C and helps provides an API which interfaces with the GPU. Hence, CUDA enables writing to as acceleration, offloading computationally expensive work to the GPU which has thousands of core, not multiple like CPU does.
When assigning work to GPU, it is often assigned in a grid of clocks and blocks of threads. Due to the fact that memory must be copied to and from the device (GPU) this happens to slow down some operations and introduces difficulties while programming. In addition, it is important to understand that cores in a block run in lockstep. Which means opertions such as jumps permitted by highly discouraged.

We will use CUDA C which is essentially C/C++ with a few extensions that allows one to execute functions on the GPU using many threads in paralle.

# Terminology
When we talk about blocks, threads, and grids we are simply talking about ways to split up work which is to be processed by the GPU.

In terms of the hierarchy, threads make up a block and blocks make up a grid. A grid would execute on the GPU which is composed of many multi-processors. Every multiprocessor is responsible fo rexecuting one or more of the blocks in the grid. Multiprocessors consists of many stream processors, which is then responsible for running one or more of the threads in the block.

If we are going to talk about the purpose of grids and blocks, these units are important to undestand so maximum utilization can be acheieved of the cores on the GPU. Each card is different in its architecture such as the number of multiprocessors and the number of stream processors per multiprocessor), hence it is very importnant to know what architecture someone is targeting or have programmatic means of determining how many threads to execute per block.

# What is a kernel?
Kernel is a function that is invoked by the host (CPU) and executed on the device (GPU). They are simultaneously executed accross potentially hundreds or thousands of cores at once, and must use the provided variables to determine their unique range of memoty and mutate.

# Syntax
CUDA source code is stored in .cu files

It is compiled using nvcc 

nvcc is the NVIDIA CUDA Compiler

__ : GPU special keywords have (underscore underscore) before them which shows it is a kernel that runs on the device so it runs on the GPU. But, you call it from the host and launches this function on the device

__global__ specifies the function is a kernel, which means it is invoked by the CPU and executed on the GPU

__device__ specifies the function is invoked by and executed on the GPU

__host__ specifies the function is invoked by and executed on the CPU

kernel<<<blocks,threads>>>();  syntax way in calling the kernel, this tells the compiler something about the way parallelism is built, such as it is built in two layers: layers of thread and layer of block


# First CUDA Program

load the CUDA module on the node: module load apps/cuda/7.5

run executable name to tell you all about the card: deviceQuery

text editor: vi

name of the file: helloworld.cu

to exist the file: esc : wq

compile: nvcc helloworld.cu -o hello

run: ./hello

----------------------
##Example 1:

        #include <stdio.h>

        //mykernel: kernel that runs in parallel on the CPU

        __global__ void mykernel(void){
        }

        //main: host code

        int main(void){

            mykernel<<<1,1>>>(); 
        
            printf("Hello  World!\n");
        
            return 0;
       
        }

-----------------------
##Example 2:

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


## Summary and Conclusions

Here are only a few extensions to C required to “port” a C code to CUDA C: the __global__ declaration specifier for device kernel functions; the execution configuration used when launching a kernel; and the built-in device variables blockDim, blockIdx, and threadIdx used to identify and differentiate GPU threads that execute the kernel in parallel.

One advantage of the heterogeneous CUDA programming model is that porting an existing code from C to CUDA C can be done incrementally, one kernel at a time.



References:

1)https://medium.com/@timer150/introduction-to-cuda-c038b85c35ba

2) https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/#:~:text=In%20CUDA%2C%20the%20host%20refers,functions%20executed%20on%20the%20device.


-----------------------------------------

## Project 1 
This  project  is  about  computing  spatial  distance  histogram  (SDH)  of  a  collection  of  3D  points.The SDH problem can be formally described as follows:  given the coordinates ofNparticles (e.g.,atoms, stars, moving objects in different applications) and a user-defined distancew, we need tocompute the number of particle-to-particle distances falling into a series of ranges (named buckets)of  widthw:  [0, w),[w,2w), . . . ,[(l−1)w, lw].   Essentially,  the  SDH  provides  an  ordered  list  ofnon-negative integersH= (h0, h1, . . . , hl−1), where eachhi(0≤i < l) is the number of distancesfalling into the bucket [iw,(i+ 1)w).  Clearly, the bucket widthwis a key parameter of an SDH tobe computed.2  Getting StartedFor you to get started, we have written a C program to compute SDH. The attached fileSDH.cushows a sample program for CPUs and serves as the starting point of your coding in this project.Interesting thing is that you can still compile and run it under the CUDA environment.  Specifically,you can type the following command to compile it.nvcc SDH.cu -o SDHTo run the code, you add the following line to your testing script file:./SDH 10000 500.0Note that the executable takes two arguments, the first one is the total number of data points andthe second one is the bucket width (w) in the histogram to be computed.  We strongly suggest youcompile and run this piece of code before you start coding.3  Tasks to PerformYou have to write a CUDA program to implement the same functionality as shown in the CPUprogram.  Both CUDA kernel function and CPU function results should be displayed as output.Write a CUDA kernel that computes the distance counts in all buckets of the SDH, by makingeach thread work on one input data point.  After all the points are processed, the resulting SDHshould be copied back to a host data structure.

1.  Transfer the input data array (i.e., atomlist as shown in the sample code) onto GPU deviceusing CUDA functions.

2.  Write a kernel function to compute the distance between one point to all other points andupdate the histogram accordingly. Note that between any two points there should be onlyone distance counted.

3.  Copy the final histogram from the GPU back to the host side and output them in the sameformat as we did for the CPU results.

4.  Compare this histogram with the one you computed using CPU bucket by bucket.  Outputany differences between them - this should be done by printing out a histogram with the samenumber of buckets as the original ones except each bucket contains the difference in countsof the corresponding buckets.Note that the output of your program should only contain the following: 

        (1) the two histograms,one computed by CPU only and the other computed by your GPU kernel. 
        
        (2) any difference youfound between these two histograms.4  EnvironmentAll projects will be tested on CUDA 7.5 on one machine of the C4 lab.

If you prefer to work on yourown computer, make sure your project can be executed on the GPU card of a C4 lab computer.

5.  Instructions to Submit ProjectYou should submit one .cu file with your implementation.  For this project,  we suggest you justadd code to the file named SDH.cu we provided.  Rename the file asproj1-xxx.cu, wherexxxisyour USF NetID. Submit the file only via assignment link in Canvas.  E-mail or any other form ofsubmission will not be graded.  Once you submit your file to Canvas, try to download the submittedfile and open it in your machine to make sure no data transmission problems occurred.  For that,we suggest you finish the project and submit the file at lease one hour before the deadline.

6.  Rules for GradingFollowing are some of the rules that you should keep in mind while submitting the project.  Theproject will be graded by a set of test cases we run against your code.
        •All programs that fail to compile will get zero points.  We expect your submission be compiledby running the simple line of command shown above.
        •If, after the submission deadline, any changes are to be made to make the main code work,they should be done within 3 lines of code.  This will incur a 30% penalty.
        •Program should run for different numbers of CUDA blocks and threads.  Use of any tricks tomake it run on one thread or only on CPU will result in zero points.  However, performanceof your GPU code is not considered in grading


----------
## Project 2 
GPU Project 2:  A Better Version of SDH Computing Program Course: CIS6930 Massive Parallel Computing 

OverviewIn this project you will measure the performance of GPU memory system,  perform different ex-periments,  and write a report on your findings.  The problem in hand is:  computing the spatialdistance histogram of a set of points.  This is the same problem that you worked on in Project 1.The main objective of this project is to write very efficient CUDA code,  unlike in Project 1where performance was not a grading criterion.  You have the freedom to choose any techniquesyou  learned  from  class  to  improve  your  program,  these  include  (but  are  not  limited  to):  usingshared memory to hold input data, manipulating input data layout for coalesced memory access,managing thread workload to reduce code divergence, atomic operations, private copies of output,and shuffle instructions.  You can use a combination of different techniques, and the code will begraded based on your rank in a class-wise contest, in which we will use a set of different datasets totest your code.  Therefore, your goal is to try all you can think of to optimize your program.  Thatsaid, we will make your job easier by introducing the following paper published by our group.Napath Pitaksirianan, Zhila Nouri, and Yi-Cheng Tu.  “Efficient 2-Body Statistics Computationon  GPUs:  Parallelization  &  Beyond”.  Proceedings of 45th International Conference on ParallelProcessing, pp.  380-385., August 2016.The  paper  describes  a  series  of  techniques  for  achieving  high  performance  in  dealing  with  agroup of problems that share similar computing patterns as the SDH problem.2  Tasks to PerformWrite a CUDA program to implement the same functionalities as required in Project 1, performdifferent  experiments,  and  write  a  short  report  about  your  project.   The  CUDA  kernel  functionresults and running time of the kernel(s) should be displayed as output.  Thus, your main task it towrite an efficient CUDA kernel to compute the SDH. In addition, your program should also includethe following features.Input/Output of Your Program:You have to modify the program to take a different numberof command line arguments from what you did in Project 
1.  Your program should take care of bador missing inputs and throw appropriate errors in a reasonable way.  In particular, here is what weexpect in launching your program:./proj2 {#of_samples} {bucket_width} {block_size}1
whereproj2is assumed to be the executable after compiling your project.  The first two argumentsare the same as in Project 1, while the last one is the number of threads within each block yourCUDA kernel should be launched.The output of your program should print out the SDH you computed as in Project 1.  Followingthe SDH, you should add a line to report the performance of your kernel, it should look like thefollowing sample.

******** Total Running Time of Kernel = 2.0043 sec 

*******Please read Section 3 for details of measuring kernel running time.

Project ReportWrite a report to explain clearly how you implemented the GPU kernels, with a focus on what techniques you used to optimize performance.3  Measuring Running TimeThe running time of the CPU implementation can be measured using different time functions avail-able in the C programming libraries.  One such function useful in measuring time isgettimeofday.

Another is therdtscinstruction supported on Pentium  CPUs. You  can  also use theclock()function to record the time. However, these functions cannot be used to measure the running timeof  GPU  kernels.There are special event functions that  are  used to record the  running time ofkernels. Following is an example to record running time of a kernel.

1:  cudaEvent_t start, stop;

2:  cudaEventCreate(&start);

3:  cudaEventCreate(&stop);

4:  cudaEventRecord( start, 0 );

5:  /* Your Kernel call goes here */

6:  cudaEventRecord( stop, 0 );

7:  cudaEventSynchronize( stop );

8:  float elapsedTime;

9:  cudaEventElapsedTime( &elapsedTime, start, stop );

10: printf( "Time to generate: %0.5f ms\n", elapsedTime );

11: cudaEventDestroy( start );

12: cudaEventDestroy( stop );

Aneventin CUDA is essentially a GPU time stamp that is recorded at a user specified pointin time.  The API is relatively easy to use, since taking a time stamp consists of just two steps:creating  an  event  and  subsequently  recording  an  event. For example, at the beginning  of  somesequence of code, we instruct the CUDA runtime to make a record of the current time.  We do soby creating and then recording the event (lines 2−4).  The exact nature of second argument in line4 is unimportant for our purposes right now (use 0 always).To time a block of code, we will want to create both a start event and a stop event.  We willrecord the CUDA time when we tell it to do some work on the GPU and then record the time againwhen we’ve stopped.
Unfortunately, there is still a problem with timing GPU code in this way.  The fix will requireonly one line of code but will require some explanation.  The trickiest part of using events arises asa consequence of the fact that some of the calls we make in CUDA C are actually asynchronous.For example,  when we launch the kernel in line 5,  the GPU begins executing our code,  but theCPU continues executing the next line of our program before the GPU finishes.  This is excellentfrom a performance standpoint because it means we can be computing something on the GPU andCPU at the same time, but conceptually it makes timing tricky.You should imagine calls tocudaEventRecord()as an instruction to record the current timebeing  placed  into  the  GPU’s  pending  queue  of  work. As a result,  our  event  won’t  actually  berecorded until the GPU finishes everything prior to the call tocudaEventRecord().  In terms ofhaving our stop event measure the correct time,  this is precisely what we want.  But we cannotsafely read the value of the stop event until the GPU has completed its prior work and recordedthe stop event.  Fortunately, we have a way to instruct the CPU to synchronize on an event, theevent API functioncudaEventSynchronize().Now, we have instructed the runtime to block further instruction (line 7) until the GPU hasreached the stop event.  When the call tocudaEventSynchronize()returns, we know that all GPUwork before the stop event has completed, so it is safe to read the time stamp recorded in stop.  It isworth noting that because CUDA events get implemented directly on the GPU, they are unsuitablefor timing mixtures of device and host code.  That is, you will get unreliable results if you attemptto use CUDA events to time more than kernel executions and memory copies involving the device.The functioncudaEventElapsedTime()is a utility that computes the elapsed time between twopreviously recorded events. The time in milliseconds elapsed between the two events is returned inthe first argument, the address of a floating-point variable.The call tocudaEventDestroy()needs to be made when we’re finished using an event createdwithcudaEventCreate().  This is identical to callingfree()on memory previously allocated withmalloc(), so we needn’t stress how important  it is to match  everycudaEventCreate()with acudaEventDestroy().

The content about measuring time on GPUs explained in this section is taken from the followingreference book:Jason Sanders and Edward Kandrot.  “CUDA  by  Example:  An  Introduction  to  General-PurposeGPU Programming”.  Addison-Wesley, 2011.
