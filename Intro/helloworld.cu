
#include <stdio.h>


__global__
void mykernel(void){
}


//main
int main(void){
	mykernel<<<1,1>>>(); // parallelism (1 block and 1 thread) Only thing required to run on the GPU
	printf("Hello  World!\n");
	return 0;

}
