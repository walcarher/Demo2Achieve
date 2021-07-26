import sys
import torch
import chimera_lib 

# FPGA communication function declaration
class FPGA_COMM(torch.autograd.Function):
    @staticmethod
    def open():
        if chimera_lib.open():
            sys.exit("FPGA device could not be opened")
            
    @staticmethod
    def close():
        chimera_lib.close()
        
    @staticmethod
    def write(input):
        chimera_lib.write(input)
        
    @staticmethod
    def read():
        chimera_lib.read()

# Function call   
comm = FPGA_COMM()
# Open FPGA Device
comm.open()
# Start random Integer tensor (Range 0x0 to max signed integer value 0x7FFFFFFF)
input = torch.randint(2147483647,(8,16,16), dtype = torch.int32)
#input = torch.ones([8,16,16], dtype = torch.int32)
print(input.size())
# Write tensor to On-Chip memory on FPGA
comm.write(input)
# Closing device and freeing memory
comm.close()