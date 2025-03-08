import time
from controls import Controls
#use this to kill car throttle if other programs break it

rover = Controls()
rover.forward(10)
time.sleep(.01)

rover.brake()
rover.turn_center()