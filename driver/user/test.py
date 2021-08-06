import sys
import torch
import math
import time
# Importiong CHIMERA module for comuunication between TX2 and C10GX
import chimera_lib 

def init_tensor(tensor):
    i = 0
    C = tensor.size(1)
    H = tensor.size(2)
    W = tensor.size(3)
    for h in range(H):
        for w in range(W):
            for c in range(C):
                tensor[0][c][h][w] = i
                i += 1
                if i == 256:
                    i = 0
    return tensor

print("Int32 tensor transfer test (chunks of 2048 DWords)")

# FPGA communication function declaration
class fpga_comm(torch.nn.Module):
    @staticmethod
    def open():
        if chimera_lib.open():
            sys.exit("FPGA device could not be opened")
            
    @staticmethod
    def close():
        chimera_lib.close()
        
    @staticmethod
    def quantize(input):
        output = chimera_lib.read(input)
        return output
        
    @staticmethod
    def write(input):
        chimera_lib.write(input)
        
    @staticmethod
    def read(input):
        output = chimera_lib.read(input)
        return output

# Function call   
comm = fpga_comm()
# Open FPGA Device
comm.open()
# Start empty 32-bit Integer tensor
input = torch.zeros((1,8,16,16), dtype = torch.int32, device = "cuda")
# Start input tensor with an increasing sequence from 0 up to 255
# for C*H*W/256 times starting with dimension 1, then 2 and finally 3
input = init_tensor(input)
# Quantize and pack DWORDs in a single 32b Integer (4 DWORDs with 4 UInt8 per address)
# Tensor depth (channel) dimension is reduced by 4 
print("Quantize tensor to Int8 and compress into a Int32 Tensor with C/4")
start = time.time()
quantized_input = chimera_lib.quantize(input)
elapsed = time.time() - start
print("Quantization elapsed time:", elapsed*1000, " ms")
# Write tensor to On-Chip memory on FPGA
print("Write Int32 tensor sequence with values from 0 to 255 for CHW/256 times")
start = time.time()
comm.write(quantized_input)
elapsed = time.time() - start
print("Write elapsed time:", elapsed*1000, " ms")
# Initialize an zeros ouput tensor with a given dimension and size to be read
output = torch.zeros((1,8,16,16), dtype = torch.int32, device = "cuda")
# Read tensor from On-Chip memory from FPGA as Integer32
print("Read Int32 tensor with a sequence from 0-255")
start = time.time()
output = comm.read(output)
elapsed = time.time() - start
print("Read elapsed time:", elapsed*1000, " ms")
# Closing device and freeing memory
comm.close()