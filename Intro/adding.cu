/*
*
* Addition on the Device: add()
*
*/



#include <iostream>
#include <math.h>

__global__
void add(int *a, int *b, int *c)
{
	*c = *a + *b;
}

//main
int main(void){
	int a, b, c;  //host copies of a,b,c
	int *d_a, *d_b, *d_c; //device copies of a,b,c
	int size = sizeof(int);

	//allocating space for device copies of a,b,c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	//setup input values: initializing our input data
	a = 2;
	b = 7;

	//copy inputs to device
	cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

	//launch add() kernel in GPU: pass arrgument like a normal function
	add<<<1,1>>>(d_a, d_b, d_c);

	//copy result back to host
	cudaMemcpy (&c, d_c, size, cudaMemcpyDeviceToHost);

	//cleanup
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;

}

