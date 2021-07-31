import sys
import torch
import time
# Importiong CHIMERA module for comuunication between TX2 and C10GX
import chimera_lib 

def init_tensor(tensor):
    C = tensor.size(1)
    H = tensor.size(2)
    W = tensor.size(3)
    for h in range(H):
        for w in range(W):
            for c in range(C):
                tensor[0][c][h][w] = c+w*C+h*W*C
    return tensor

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
    def quantize(input):
        output = chimera_lib.quantize(input)
        return output
        
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
# Start empty 32-bit Integer tensor
input = torch.empty((1,8,16,16), dtype = torch.int32, device = "cpu")
# Start input tensor with a sequence from 0-C*H*W starting with dimension C, then H and finally W
input = init_tensor(input)
# Quantize and pack DWORDs in a single 32b Integer (4 DWORDs per address)
# Tensor depth (channel) dimension is reduced by 4 
print("Quantize Int32 tensor to Int8 and compress it to Int32 with C/4")
start = time.time()
quantized_input = comm.quantize(input)
elapsed = time.time() - start
print("Quantization elapsed time:", elapsed*1000, " ms")
# Write tensor to On-Chip memory on FPGA
print("Write Int32 tensor sequence with values from 0-C*H*W-1")
start = time.time()
comm.write(quantized_input)
elapsed = time.time() - start
print("Write elapsed time:", elapsed*1000, " ms")
# Initialize empty tensor with a given dimension and size
output = torch.empty((1,8,16,16), dtype = torch.int32, device = "cuda")
# Read tensor from On-Chip memory from FPGA as Integer32
#print("Read Int32 tensor with values0x7FFFFFFF")
start = time.time()
output = comm.read(output)
elapsed = time.time() - start
print("Read elapsed time:", elapsed*1000, " ms")
print(output)
# Closing device and freeing memory
comm.close()