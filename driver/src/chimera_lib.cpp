#include <torch/extension.h>
#include <iostream>

extern "C" {
   #include "chimera_lib.h"
   #include "../altera_dma_cmd.h"
}

int write_to_fpga(torch::Tensor torch_tensor)
{
	int *tensor = (int*)malloc(((struct dma_status *)buf)->altera_dma_num_dwords*sizeof(int));
	auto tensor_acc = torch_tensor.accessor<float,3>();
	int H = tensor_acc.size(0);
	int W = tensor_acc.size(1);
	int C = tensor_acc.size(2);
	int h, w, c;
	
	for (h = 0; h < H; h++) {
		for (w = 0; w < W; w++) {
			for (c = 0; c < C; c++) {
			tensor[h*w*c] = (int)tensor_acc[h][w][c];
			}
		}
	}
	write_to_fpga_raw(tensor);
	free(tensor);
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