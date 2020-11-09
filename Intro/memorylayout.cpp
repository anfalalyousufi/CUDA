


#include <stdio.h>

int global_initialized = 21;
int global_uninitialized;

int func(){
	int local_inside_func;

	int *pointer = malloc(sizeof(int));
	free(pointer);
}

int main(int argc, char *argv[]){
	int local_inside_main;

	func();
	return 0;
}
