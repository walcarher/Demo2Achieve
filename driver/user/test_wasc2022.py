import sys
from torch import nn
import torch
import math
import time
import numpy as np
# Importiong CHIMERA module for comuunication between TX2 and C10GX
import chimera_lib 
# Importing OpenCV for Computer Vision and Image Processing
import cv2

print("Heterogeneous Sobel Filtering: Task partitioning with Tiling")

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
        
# Sobel module in X direction for GPU
class SobelX(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)
        Sx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]], device="cuda")
        Sx = Sx.unsqueeze(0).unsqueeze(0)
        self.filter.weight = nn.Parameter(Sx, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x
        
# Sobel module in Y direction for GPU
class SobelY(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)
        Sy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]], device="cuda")
        Sy = Sy.unsqueeze(0).unsqueeze(0)
        self.filter.weight = nn.Parameter(Sy, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x

def test():
    #cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080,format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1")
    cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1")
    if cap.isOpened():
        # Image resolution (VGA)
        width = 640
        height = 480
    	# Window creation and specifications (1920x1080)
        W = 1920
        H = 1080
        font = cv2.FONT_HERSHEY_PLAIN
        # for CPU
        windowNameCPU = "RGB Image on CPU"
        cv2.namedWindow(windowNameCPU, cv2.WINDOW_NORMAL)
        cv2.moveWindow(windowNameCPU,0,0)
        cv2.resizeWindow(windowNameCPU,int(W/2),int(H/3)-50)
        cv2.setWindowTitle(windowNameCPU, "RGB Image on CPU")
        # for GPU
        windowNameGPU = "Sobel (X direction) on GPU"
        cv2.namedWindow(windowNameGPU, cv2.WINDOW_NORMAL)
        cv2.moveWindow(windowNameGPU,0,int(H/3)+15)
        cv2.resizeWindow(windowNameGPU,int(W/2),int(H/3)-50)
        cv2.setWindowTitle(windowNameGPU, "Sobel (X direction) on GPU")
        # for FPGA
        windowNameFPGA = "Sobel (Y direction) on FPGA"
        cv2.namedWindow(windowNameFPGA, cv2.WINDOW_NORMAL)
        cv2.moveWindow(windowNameFPGA,0,int(2*H/3))
        cv2.resizeWindow(windowNameFPGA,int(W/2),int(H/3)-50)
        cv2.setWindowTitle(windowNameFPGA,"Sobel (Y direction) on FPGA")
        start = 0.0
        end = 0.0
    else:
        print("Unable to open camera")
        exit(-1)
    # Model initialization
    sobelGPU = SobelX()
    sobelFPGA = SobelY()
    # Tile Size of 32
    tile_size = 32
    # Function call for FPGA communication object   
    comm = fpga_comm()
    # Open FPGA Device
    comm.open()
    # Initilize output tensor
    output = torch.zeros((1,1,tile_size,tile_size), dtype = torch.int32, device = "cpu")
    # Get frame
    print("Capturing frame...")
    res, imgCPU = cap.read()
    if res:
        # Image object initialization
        imgCPU = cv2.resize(imgCPU, (width,height))
        imgFPGA = cv2.cvtColor(imgCPU, cv2.COLOR_BGR2GRAY)
        imgGPU = cv2.cvtColor(imgCPU, cv2.COLOR_BGR2GRAY)
        # GPU SobelX
        inputGPU = torch.from_numpy(imgGPU)
        inputGPU = inputGPU.to(dtype = torch.float, device = "cuda").unsqueeze(0).unsqueeze(0)
        inputGPU2 = inputGPU
        for x in range(300):
            inputGPU = sobelGPU(inputGPU2)
        imgGPU = np.uint8(inputGPU.cpu().numpy().squeeze(0).squeeze(0))
        # FPGA Tiling Partitioning, Processing and Communication
        inputFPGA = torch.from_numpy(imgFPGA)
        inputFPGA = inputFPGA.to(dtype = torch.int32, device = "cpu").unsqueeze(0).unsqueeze(0)
        for y in range(int(height/tile_size)):
            for x in range(int(width/tile_size)):
                comm.write(inputFPGA[:,:,y*tile_size:(y*tile_size+tile_size),x*tile_size:(x*tile_size+tile_size)])
                output = comm.read(output)
                imgFPGA[y*tile_size:(y*tile_size+tile_size),x*tile_size:(x*tile_size+tile_size)] = np.uint8(output.numpy().squeeze(0).squeeze(0))
        inputFPGA = inputFPGA.to(dtype = torch.float, device = "cuda")
        inputFPGA = sobelFPGA(inputFPGA)
        imgFPGA = np.uint8(inputFPGA.cpu().numpy().squeeze(0).squeeze(0))
        # Showing results
        # for CPU
        imgCPU = cv2.putText(imgCPU,"CPU",(10,75),font,5,(0,0,255),5,2)
        cv2.imshow(windowNameCPU, imgCPU)
        # for GPU
        imgGPU = cv2.cvtColor(imgGPU, cv2.COLOR_GRAY2BGR)
        imgGPU = cv2.putText(imgGPU,"GPU",(10,75),font,5,(0,255,0),5,2)
        cv2.imshow(windowNameGPU, imgGPU)
        # for FPGA
        imgFPGA = cv2.cvtColor(imgFPGA, cv2.COLOR_GRAY2BGR)
        imgFPGA = cv2.putText(imgFPGA,"FPGA",(10,75),font,5,(255,0,0),5,2)
        cv2.imshow(windowNameFPGA, imgFPGA)
        while True:
            key = cv2.waitKey(0)
            if key == 27:
                cv2.destroyAllWindows()
                comm.close()
                break
    else:
         print("Unable to read image")
         comm.close()
         exit(-1) 

############################################
if __name__ == '__main__':
    test()