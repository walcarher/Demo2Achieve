import sys
import torch
import math
import time
import numpy as np
# Importiong CHIMERA module for comuunication between TX2 and C10GX
import chimera_lib 
# Importing OpenCV for Computer Vision and Image Processing
import cv2

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

def test():
    #cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080,format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1")
    cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1")
    if cap.isOpened():
    	# Window creation and specifications
        windowName = "OpenCV Camera Test"
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.moveWindow(windowName,0,0)
        cv2.resizeWindow(windowName,640,480)
        cv2.setWindowTitle(windowName,"OpenCV Camera Test")
        font = cv2.FONT_HERSHEY_PLAIN
        helpText="'Esc' to Quit"
        showFullScreen = False
        showHelp = False
        start = 0.0
        end = 0.0
    else:
        print("Unable to open camera")
        exit(-1)
        
    # Function call for FPGA communication object   
    comm = fpga_comm()
    # Open FPGA Device
    comm.open()
    # Initilize output tensor
    output = torch.zeros((1,1,32,32), dtype = torch.int32, device = "cpu")

    while True:
        res, img = cap.read()
        if res:
            img = cv2.resize(img, (32,32))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            input = torch.from_numpy(img)
            input = input.to(dtype = torch.int32, device = "cpu").unsqueeze(0).unsqueeze(0)
            comm.write(input)
            output = comm.read(output)
            img = np.uint8(output.numpy().squeeze(0).squeeze(0))
            if showHelp == True:
                cv2.putText(img, helpText, (11,20), font, 1.0, (32,32,32), 4, cv2.LINE_AA)
                cv2.putText(img, helpText, (10,20), font, 1.0, (240,240,240), 1, cv2.LINE_AA)
            end = time.time()
            cv2.putText(img, "{0:.0f}fps".format(1/(end-start)), (531,50), font, 3.0, (32,32,32), 8, cv2.LINE_AA)
            cv2.putText(img, "{0:.0f}fps".format(1/(end-start)), (530,50), font, 3.0, (240,240,240), 2, cv2.LINE_AA)
            cv2.imshow(windowName, img)
            start = time.time()
            key = cv2.waitKey(1)
            if key == 27: # Check for ESC key
                cv2.destroyAllWindows()
                comm.close()
                break;
            elif key==74: # Toggle fullscreen; This is the F3 key on this particular keyboard
                # Toggle full screen mode
                if showFullScreen == False : 
                    cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL) 
                    showFullScreen = not showFullScreen
        else:
             print("Unable to read image")
             comm.close()
             exit(-1) 

############################################
if __name__ == '__main__':
    test()