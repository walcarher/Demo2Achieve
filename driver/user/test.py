import sys
import torch
import time
# Importiong CHIMERA module for comuunication between TX2 and C10GX
import chimera_lib 

print("Int32 tensor transfer test (chunks of 2048 DWords)")

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
    def read(output):
        output = chimera_lib.read(output)
        return output

# Function call   
comm = FPGA_COMM()
# Open FPGA Device
comm.open()
#for i in range (100):
# Start random Integer tensor (Range 0x0 to max signed integer value 0x7FFFFFFF or 2147483647d)
input = torch.randint(2147483647,(1,8,16,16), dtype = torch.int32, device = "cuda")
# Write tensor to On-Chip memory on FPGA
print("Write random Int32 tensor with values between 0-0x7FFFFFFF")
start = time.time()
comm.write(input)
elapsed = time.time() - start
print("Write elapsed time:", elapsed*1000, " ms")
# Initialize empty tensor with a given dimension and size
output = torch.empty((1,8,16,16), dtype = torch.int32, device = "cpu")
# Read tensor from On-Chip memory from FPGA as Integer32
print("Read Int32 tensor with values0x7FFFFFFF")
start = time.time()
output = comm.read(output)
elapsed = time.time() - start
print("Read elapsed time:", elapsed*1000, " ms")
#print(output)
# Closing device and freeing memory
comm.close()