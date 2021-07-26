#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include "../altera_dma_cmd.h"

int open_fpga()
{
	f = open ("/dev/altera_dma", O_RDWR);
	if (f == -1) {
        printf ("Couldn't open the FPGA device.\n");
        return 1;
    } else {
        printf ("Opened the device: file handle #%lu!\n", (long unsigned int)f);
		*buf = malloc(sizeof(struct dma_status));
		cmd.usr_buf_size = sizeof(struct dma_status);
		return 0;
	}
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

int close_fpga()
{
	close (f);
	return 0;
}