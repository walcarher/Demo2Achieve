extern ssize_t f;
extern char *buf;
extern struct dma_cmd cmd;

int open_fpga();
int close_fpga();
int write_to_fpga_raw(int *tensor);
int read_from_fpga_raw(int *tensor);