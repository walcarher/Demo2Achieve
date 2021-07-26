#include <torch/extension.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include "../altera_dma_cmd.h"

ssize_t f;
char *buf;
struct dma_cmd cmd;
int *tensor;

int open_fpga()
{
	f = open ("/dev/altera_dma", O_RDWR);
	if (f == -1) {
        printf ("Couldn't open the FPGA device.\n");
        return 1;
    } else {
        printf ("FPGA device successfully opened: file handle #%lu!\n", (long unsigned int)f);
		buf = (char*)malloc(sizeof(struct dma_status));
		cmd.usr_buf_size = sizeof(struct dma_status);
		tensor = (int*)malloc(((struct dma_status *)buf)->altera_dma_num_dwords*sizeof(int));
		return 0;
	}
}

int close_fpga()
{
	free(tensor);
	free(buf);
	close (f);
	return 0;
}

int write_to_fpga_raw(int *tensor)
{
	// Reads Tensor from CPU/GPU to FPGA
	ioctl(f, ALTERA_IOCX_READ_TENSOR, tensor);
	ioctl(f, ALTERA_IOCX_WAIT);
	cmd.cmd = ALTERA_CMD_READ_STATUS;
	cmd.buf = buf;
	write(f, &cmd, 0);
	return 0;
}

int read_from_fpga_raw(int *tensor)
{
	// Writes Tensor from FPGA to CPU/GPU
	ioctl(f, ALTERA_IOCX_WRITE_TENSOR, tensor);
	ioctl(f, ALTERA_IOCX_WAIT);
	cmd.cmd = ALTERA_CMD_READ_STATUS;
	cmd.buf = buf;
	write (f, &cmd, 0);
	return 0;
}

int init_tensor(int length, int *tensor)
{
	int i = 0;
	for (i = 0; i < length; i++) {
		tensor[i] = i;
		//tensor[i] = 4294967295;
	}
	return 0;
}

int write_to_fpga(torch::Tensor torch_tensor)
{
	auto tensor_acc = torch_tensor.accessor<int,3>();
	int C = tensor_acc.size(0);
	int H = tensor_acc.size(1);
	int W = tensor_acc.size(2);
	int h, w, c;
	
	for (h = 0; h < H; h++) {
		for (w = 0; w < W; w++) {
			for (c = 0; c < C; c++) {
			// Index unrolling
			tensor[c+w*C+h*C*W] = tensor_acc[c][h][w];
			}
		}
	} 
	//init_tensor(((struct dma_status *)buf)->altera_dma_num_dwords, tensor);
	write_to_fpga_raw(tensor);
	return 0;
}

int read_from_fpga(torch::Tensor tensor)
{
	return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("open", &open_fpga, "Open FPGA device");
	m.def("close", &close_fpga, "Close FPGA device");
	m.def("write", &write_to_fpga, "Tensor write to FPGA");
	m.def("read", &read_from_fpga, "Tensor read from FPGA");
}