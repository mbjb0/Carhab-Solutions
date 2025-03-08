controls: PWM code to interface with adafruit contoller
FSM main: main code for self driving
FSM_instruct: driving logic state machine
FSM_simulation: pygame simulator to develop FSM_instruct
FSM_visualization: state visualizer for the driving logic state machine. Called in FSM main and FSM_simulation
sort: algorithm for assigning unique ids to trackers which I DID NOT WRITE. https://github.com/abewley/sort


If you are trying to replicate this project I would use this code as a reference point, but do not try to directly build on top of it
It is very fragile, redundant and unoptimized.

This code must run on a windows machine. If you are using linux, be prepared to change quite a bit.
This was run on a gaming laptop with a 3070, you will need a different yolo model if you want to run on an edge ai device with
lower power, such as a TX2.

How to start a project based off this code (mostly for EEC 174 students):
Look for any file paths and change them to match your own. All file paths are direct instead of relative in this project.
Use controls_test to confirm that controls.py works correctly. 
Then write a small main file that just gets frames and calls classification.py
functions to confirm those work.
At that point you should be good to start modifying the actual FSM_main and FSM_instruct.

FSM_visualization and FSM_simulation will be the hardest to modify. These were made for the debugging and development of
this specific project, they will likely not be very useful to you.