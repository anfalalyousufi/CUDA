/*
   ==================================================================
        Programmer: Yicheng Tu (ytu@cse.usf.edu)
        The basic SDH algorithm implementation for 3D data
        To compile: nvcc SDH.c -o SDH in the rc machines
   ==================================================================
*/

/*********************************************************************
By      : Anfal AlYousufi
Course  : CIS 6930- Programming Massively Parallel Systems
Project : 1
Date    : May 26th 2016

Summer 2018
*********************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define BOX_SIZE    23000 /* size of the data box on one dimension            */

/* descriptors for single atom in the tree */
typedef struct atomdesc {
    double x_pos;
    double y_pos;
    double z_pos;
} atom;

typedef struct hist_entry{
    //float min;
    //float max;
    unsigned long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;

bucket * histogram;     /* list of all buckets in the histogram   */
long long   PDH_acnt;   /* total number of data points            */
int num_buckets;        /* total number of buckets in the histogram */
double   PDH_res;       /* value of w                             */
atom * atom_list;       /* list of all data points                */


/* These are for an old way of tracking time */
struct timezone Idunno; 
struct timeval startTime, endTime;


//
  //  Checking for CUDA Error
//
void checkError(cudaError_t e, const char out[]){
    if(e != cudaSuccess){
        printf("There is a CUDA Error: %s, %s \n", out, cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }
}
// 
  //  distance of two points in the atom_list 
//
__device__ 
double p2p_distance(atom *l, int ind1, int ind2) {
    
    double x1 = l[ind1].x_pos;
    double x2 = l[ind2].x_pos;
    double y1 = l[ind1].y_pos;
    double y2 = l[ind2].y_pos;
    double z1 = l[ind1].z_pos;
    double z2 = l[ind2].z_pos;
        
    return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}

//
    //SDH solution in a single CPU thread 
//
__global__ 
void PDH_baseline(bucket *histogram_in, atom *list, double width, int size) {
    int i, j, h_pos;
    double dist;
    
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = i + 1;
    for(int x = j; x < size; x++){
        dist = p2p_distance(list,i,x);
        h_pos = (int) (dist/ width);
        atomicAdd( &histogram_in[h_pos].d_cnt, 1);
    }
}
//
  //  set a checkpoint 
//	and 
  // show running time in seconds 
//
double report_running_time() {
    long sec_diff, usec_diff;
    gettimeofday(&endTime, &Idunno);
    sec_diff = endTime.tv_sec - startTime.tv_sec;
    usec_diff= endTime.tv_usec-startTime.tv_usec;
    if(usec_diff < 0) {
        sec_diff --;
        usec_diff += 1000000;
    }
    printf("Running time for GPU version: %ld.%06lds\n", sec_diff, usec_diff);
    return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}
/*
    brute-force solution in a GPU thread
*/
__global__ 
void PDH2D_baseline(bucket *histogram, atom *Atomlist, double w){
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    if(i < j){
        double dist = p2p_distance(Atomlist, i, j);
        int h_pos = (int)(dist / w);
        histogram[h_pos].d_cnt++;
        printf("%d, %d : %d, %f \n", i, j, h_pos, dist);
    }
}
/* 
    print the counts in all buckets of the histogram 
*/
void output_histogram(bucket *histogram){
    int i; 
    long long total_cnt = 0;
    for(i=0; i< num_buckets; i++) {
        if(i%5 == 0) /* we print 5 buckets in a row */
            printf("\n%02d: ", i);
        printf("%15lld ", histogram[i].d_cnt);
        total_cnt += histogram[i].d_cnt;
        /* we also want to make sure the total distance count is correct */
        if(i == num_buckets - 1)    
            printf("\n T:%lld \n", total_cnt);
        else printf("| ");
    }
}
/*
    MAIN
*/
int main(int argc, char **argv)
{
    PDH_acnt = atoi(argv[1]);
    PDH_res  = atof(argv[2]);
   
    num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
 
    size_t histogramSize = sizeof(bucket)*num_buckets;
    size_t atomSize = sizeof(atom)*PDH_acnt;
   
    histogram = (bucket *)malloc(histogramSize);
    atom_list = (atom *)malloc(atomSize);
    
    srand(1);

    /* uniform distribution */
    for(int i = 0;  i < PDH_acnt; i++) {
        atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
        atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
        atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
    }
    /* Malloc Space on Device, copy to Device */
    bucket *d_histogram = NULL;
    atom *d_atom_list = NULL;

    /* Error Checks */
    checkError( cudaMalloc((void**) &d_histogram, histogramSize), "Malloc Histogram");
    checkError( cudaMalloc((void**) &d_atom_list, atomSize), "Malloc Atom List");
    checkError( cudaMemcpy(d_histogram, histogram, histogramSize, cudaMemcpyHostToDevice), "Copy Histogram to Device");
    checkError( cudaMemcpy(d_atom_list, atom_list, atomSize, cudaMemcpyHostToDevice), "Copy Atom_List to Device");
    
    /* start counting time */
    gettimeofday(&startTime, &Idunno);
  
    /* CUDA Kernel Call */
    PDH_baseline <<<ceil(PDH_acnt/32), 32 >>> (d_histogram, d_atom_list, PDH_res, PDH_acnt);
    
    /* Checks Cuda Error*/
    checkError(cudaGetLastError(), "Checking Last Error, Kernel Launch");
    checkError( cudaMemcpy(histogram, d_histogram, histogramSize, cudaMemcpyDeviceToHost), "Copy Device Histogram to Host");
    
    /* check the total running time */ 
    report_running_time();
 
    /* print out the histogram */
    output_histogram(histogram);

    /* Error Checks */
    checkError(cudaFree(d_histogram), "Free Device Histogram");
    checkError(cudaFree(d_atom_list), "Free Device Atom_List");

    /* Free Memory */
    free(histogram);
    free(atom_list);

    /* Reset */
    checkError(cudaDeviceReset(), "Device Reset");
        
    return 0;
}
