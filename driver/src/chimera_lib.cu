#include <torch/extension.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../altera_dma_cmd.h"

// Global variables
ssize_t f;
char *buf;
struct dma_cmd cmd;
int *tensor;
int *dev_ptr;
// Max number of threads per block on JetsonTX2
// On-Chip memory of 64Kb - 2048 DWORDS (32b)
const dim3 threads(1024);
const dim3 blocks(2048/threads.x);

int init_tensor(int length, int *tensor)
{
	int i = 0;
	for (i = 0; i < length; i++) {
		tensor[i] = 0;
	}
	return 0;
}

int print_tensor(int length, int *tensor)
{
	int i = 0;
	for (i = 0; i < length; i++) {
		printf("Tensor value = %d\n", tensor[i]);
	}
	return 0;	
}

int write_to_fpga_raw(int *tensor)
{
	// Reads Tensor from CPU/GPU to FPGA
	ioctl(f, ALTERA_IOCX_READ_TENSOR, tensor);
	ioctl(f, ALTERA_IOCX_WAIT);
	return 0;
}

int * read_from_fpga_raw(int *tensor)
{
	// Writes Tensor from FPGA to CPU/GPU
	ioctl(f, ALTERA_IOCX_WRITE_TENSOR, tensor);
	ioctl(f, ALTERA_IOCX_WAIT);
	return tensor;
}

int read_status(){
	cmd.cmd = ALTERA_CMD_READ_STATUS;
	cmd.buf = buf;
	write (f, &cmd, 0);
	return 0;
}

__global__ void read_from_gpu(torch::PackedTensorAccessor32<int, 1> accessor, int* tensor_ptr,int length, int offset){
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	tensor_ptr[tid] = accessor[tid+offset*length];
}

int open_fpga()
{
	f = open ("/dev/altera_dma", O_RDWR);
	if (f == -1) {
        printf ("Error: Couldn't open the FPGA device.\n");
        return 1;
    } else {
        printf ("FPGA device successfully opened: file handle #%lu!\n", (long unsigned int)f);
		struct dma_cmd cmd;
		buf = (char*)malloc(sizeof(struct dma_status));
		cmd.cmd = ALTERA_CMD_READ_STATUS;
        cmd.buf = buf;
        write (f, &cmd, 0);
		cudaHostAlloc((void **)&tensor, ((struct dma_status *)buf)->altera_dma_num_dwords*sizeof(int), cudaHostAllocMapped);
		cudaHostGetDevicePointer((void **)&dev_ptr, (void *)tensor, 0);
		init_tensor(((struct dma_status *)buf)->altera_dma_num_dwords, tensor);
		return 0;
	}
}

int close_fpga()
{
	free(buf);
	cudaFreeHost(tensor);
	cudaFree(dev_ptr);
	cudaDeviceReset();
	close (f);
	return 0;
}

int write_to_fpga(torch::Tensor torch_tensor)
{
	if (torch_tensor.dim() != 4){
		printf("Error: Only 2D-Conv supported. Tensor must be dimension 4.\n");
		return 1;
	}
	if (torch_tensor.device().is_cpu()) {
		auto tensor_acc = torch_tensor.accessor<int, 4>();
		int C = tensor_acc.size(1);
		int H = tensor_acc.size(2);
		int W = tensor_acc.size(3);
		int c, h, w;
		int last_c = 0, last_h = 0, last_w = 0;
		int addr_c = 0, addr_h = 0, addr_w = 0;
		
		for (h = 0 ; h < H; h++) {
			for (w = 0 ; w < W; w++) {
				for (c = 0 ; c < C; c++) {
					if (addr_c+addr_w*C+addr_h*C*W == ((struct dma_status *)buf)->altera_dma_num_dwords-1){
						last_c = c;
						last_h = h;
						last_w = w;
						addr_c = c - last_c;
						addr_h = h - last_h;
						addr_w = w - last_w;
						write_to_fpga_raw(tensor);
					} else {
						addr_c = c - last_c;
						addr_h = h - last_h;
						addr_w = w - last_w;
					}
					// Index unrolling
					tensor[addr_c+addr_w*C+addr_h*C*W] = tensor_acc[0][c][h][w];
				}
			}
		}
		write_to_fpga_raw(tensor);
	} else {
		int C = torch_tensor.size(1);
		int H = torch_tensor.size(2);
		int W = torch_tensor.size(3);
		// Flatten tensor with with priority on the number of channels (C) or tensor depth
		torch::Tensor temp_tensor = torch_tensor.permute({0,3,2,1}).permute({0,2,1,3}).reshape({C*H*W});
		auto tensor_acc = temp_tensor.packed_accessor32<int, 1>();
		int chunks_num = (int)ceil(C*H*W / ((struct dma_status *)buf)->altera_dma_num_dwords);
		for (int i = 0; i < chunks_num; i++){
			// Maximum number of threads per block (1024) on the TX2 Pascal arch
			// Split tensor into accesses of N blocks with 1024 threads
			read_from_gpu<<<blocks,threads>>>(tensor_acc, dev_ptr, ((struct dma_status *)buf)->altera_dma_num_dwords, i);
			//print_tensor(((struct dma_status *)buf)->altera_dma_num_dwords, tensor);
			cudaDeviceSynchronize();
			write_to_fpga_raw(tensor); 
		}
	}
	return 0;
}

torch::Tensor read_from_fpga(torch::Tensor torch_tensor)
{
	if (torch_tensor.dim() != 4){
		printf("Error: Only 2D-Conv supported. Tensor must be dimension 4.\n");
		return torch_tensor;
	}
	tensor = read_from_fpga_raw(tensor);
	//print_tensor(((struct dma_status *)buf)->altera_dma_num_dwords, tensor);
	torch_tensor = torch::from_blob(tensor, {torch_tensor.size(0),torch_tensor.size(1),torch_tensor.size(2),torch_tensor.size(3)}, torch::dtype(torch::kInt32));
	return torch_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("open", &open_fpga, "Open FPGA device");
	m.def("close", &close_fpga, "Close FPGA device");
	m.def("write", &write_to_fpga, "Tensor write to FPGA");
	m.def("read", &read_from_fpga, "Tensor read from FPGA");
}