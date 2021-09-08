import sys
import torch
import math
import time
import numpy as np
# Importiong CHIMERA module for comuunication between TX2 and C10GX
import chimera_lib 
# Importing OpenCV for Computer Vision and Image Processing
import cv2

print("Heterogeneous Image Binarization: Task partitioning with Tiling")

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

def test():
    #cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080,format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1")
    cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1")
    if cap.isOpened():
    	# Window creation and specifications
        width = 640
        height = 480
        windowNameGPU = "RGB Image on CPU/GPU"
        cv2.namedWindow(windowNameGPU, cv2.WINDOW_NORMAL)
        cv2.moveWindow(windowNameGPU,0,0)
        cv2.resizeWindow(windowNameGPU,width,height)
        cv2.setWindowTitle(windowNameGPU, "RGB Image on CPU/GPU")
        windowNameFPGA = "Binarized Image on FPGA"
        cv2.namedWindow(windowNameFPGA, cv2.WINDOW_NORMAL)
        cv2.moveWindow(windowNameFPGA,0,0)
        cv2.resizeWindow(windowNameFPGA,width,height)
        cv2.setWindowTitle(windowNameFPGA,"Binarized Image on FPGA")
        font = cv2.FONT_HERSHEY_PLAIN
        start = 0.0
        end = 0.0
    else:
        print("Unable to open camera")
        exit(-1)
    
    # Tile Size of 32
    tile_size = 32
    # Function call for FPGA communication object   
    comm = fpga_comm()
    # Open FPGA Device
    comm.open()
    # Initilize output tensor
    output = torch.zeros((1,1,tile_size,tile_size), dtype = torch.int32, device = "cpu")

    res, img = cap.read()
    if res:
        #img = cv2.resize(img, (32,32))
        img = cv2.resize(img, (width,height))
        cv2.imshow(windowNameGPU, img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        input = torch.from_numpy(img)
        input = input.to(dtype = torch.int32, device = "cpu").unsqueeze(0).unsqueeze(0)
        for y in range(int(height/tile_size)):
            for x in range(int(width/tile_size)):
                #print("Tile [",x,",",y,"]")
                comm.write(input[:,:,y*tile_size:(y*tile_size+tile_size),x*tile_size:(x*tile_size+tile_size)])
                output = comm.read(output)
                img[y*tile_size:(y*tile_size+tile_size),x*tile_size:(x*tile_size+tile_size)] = np.uint8(output.numpy().squeeze(0).squeeze(0))
        cv2.imshow(windowNameFPGA, img)
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