# A Glance at CUDA programming and cuDNN
## Cuda
#### Somple Processing Flow
![figure 1](https://i.imgur.com/ph035pi.png)
#### Hello World
```
__global__ void mykernel(void) {
}
```
* Runs on the device
* Device functions, processed by NVUDIA compiler
```
int main(void) {
	mykernel<<<1, 1>>>();
	printf("Hello World!\n");
	return 0;
}
```
* mykernel<<<1, 1>>>(); 告知device啟動1(Blocks)*1(Units)運算單元做運算
* Processed by standard host compiler e.g. gcc, cl.exe
#### Parallel Programming
* add() will execute on the device and be called from the host
* host will assign array size
```
__global__ void add(int *a, int *b, int *c) {
	*c = *a + *b;
}
```
#### Momory Menagement
* Device pointers point to GPU memory
* Host pointers point to CPU memory

## cuDNN
* high level library