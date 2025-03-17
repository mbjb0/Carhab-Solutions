#I do not know if this will work I dont have the car
import time
from jetracer.nvidia_racecar import NvidiaRacecar
'''
>> cd $HOME
>> git clone https://github.com/NVIDIA-AI-IOT/jetracer
>> cd jetracer
>> python setup.py install
'''

class JetsonController:
    def init__(self):
        #either setup the jetson GPIO (pwm type shi) or 
        #use a higher level library like jetracer
        self.car = NvidiaRacecar
    
    def move_backward(self, debug):
        print("Moving backwards...")
        if(debug != 1):
            # Set the car's steering to full left (max negative value) and stop forward/backward movement
            self.car.steering = 0.0  # No turning
            self.car.throttle = -1.0   # backward movement
            # Wait for a short time while moving
            time.sleep(1)
            # Stop the car
            self.stop_motors()
    
    def move_forward(self, debug):
        print("Moving forward...")
        if(debug != 1):
            # Set the car's steering to full left (max negative value) and stop forward/backward movement
            self.car.steering = 0.0  # No turning
            self.car.throttle = 1.0   # forward movement
            # Wait for a short time while moving
            time.sleep(1)
            # Stop the car
            self.stop_motors()
    
    def turn_left(self, debug):
        print("Turning left...")
        if(debug != 1):
            # Set the car's steering to full left (max negative value) and stop forward/backward movement
            self.car.steering = -1.0  # Full left turn
            self.car.throttle = 0.0   # No forward/backward movement
            # Wait for a short time while turning
            time.sleep(1)
            # Stop the car after turning
            self.stop_motors()
    
    def turn_right(self, debug):
        print("Turning right...")
        if(debug != 1):
            # Set the car's steering to full left (max negative value) and stop forward/backward movement
            self.car.steering = 1.0  # Full right turn
            self.car.throttle = 0.0   # No forward/backward movement
            # Wait for a short time while turning
            time.sleep(1)
            # Stop the car after turning
            self.stop_motors()
    
    def spin_in_place(self,debug, direction="left", speed=1.0, duration=2.0):
        """
        Spins the car in place by setting the steering and throttle.

        :param direction: "left" or "right", the direction to spin in place.
        :param speed: The speed of the spin, should be between -1.0 and 1.0.
        :param duration: The time in seconds for the spin to last.
        """
        print("spinning...")
        if(debug != 1):
            if direction == "left":
                print("Spinning left...")
                self.car.steering = -1.0  # Full left steering
                self.car.throttle = speed  # Forward movement to spin left
            elif direction == "right":
                print("Spinning right...")
                self.car.steering = 1.0   # Full right steering
                self.car.throttle = -speed  # Reverse movement to spin right
            else:
                print("Invalid direction. Use 'left' or 'right'.")
                return

            time.sleep(duration)  # Spin for the specified duration
            self.stop_motors()
    
    def stop_motors(self, debug):
    # Stop both throttle and steering
        print("Stopping the car...")
        if(debug != 1):
            self.car.steering = 0.0  # Center the steering
            self.car.throttle = 0.0  # Stop the car
    
    def cleanup(self):
        # Any cleanup if necessary
        print("Cleaning up resources...")
        self.stop_motors()