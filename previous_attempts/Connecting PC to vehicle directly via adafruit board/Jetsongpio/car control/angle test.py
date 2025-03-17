import cv2
import os
from classification import RoadSignDetection
import pathlib
import time
import numpy as np
#----------CHANGE Instructiontest to driving_logic to test car movement-----------#
from Instructiontest import Instruction
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  
from controls import Controls

rover = Controls()

print("left turn test")
print("10 deg")
rover.turn_left(10)
time.sleep(1)
print("20 deg")
rover.turn_left(20)
time.sleep(1)
print("30 deg")
rover.turn_left(30)
time.sleep(1)
print("40 deg")
rover.turn_left(40)
time.sleep(1)
print("50 deg")
rover.turn_left(50)
time.sleep(1)
print("60 deg")
rover.turn_left(60)
time.sleep(1)

print("right turn test")
print("10 deg")
rover.turn_right(10)
time.sleep(1)
print("20 deg")
rover.turn_right(20)
time.sleep(1)
print("30 deg")
rover.turn_right(30)
time.sleep(1)
print("40 deg")
rover.turn_right(40)
time.sleep(1)
print("50 deg")
rover.turn_right(50)
time.sleep(1)
print("60 deg")
rover.turn_right(60)
time.sleep(1)
