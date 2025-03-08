import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  
from controls import Controls

#Place the car on a stool or something when running this.
#It will accelerate for a long time, this is to test the full range of all controls.
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
rover.turn_center()

print("continuous left turn test")
degree = 0
while(degree < 60):
    current_time = time.time()
    rover.turn_left(degree)
    degree += 1
    hang_time = current_time - time.time()
    print("hang time:", hang_time)

print("continuous right turn test")
degree = 0
while(degree < 60):
    current_time = time.time()
    rover.turn_right(degree)
    degree += 1
    hang_time =  time.time() - current_time
    print("hang time:", hang_time)

print("forward slow")
rover.forwardslow()
time.sleep(2)
print('braking')
rover.brake()

print("forward 20%")
rover.forward(20)
time.sleep(2)
print('braking')
rover.brake()

print("forward 20%")
rover.forward(20)
time.sleep(2)
print('braking')
rover.brake()

print("forward 30%")
rover.forward(30)
time.sleep(2)
print('braking')
rover.brake()

print("forward 40%")
rover.forward(40)
time.sleep(2)
print('braking')
rover.brake()

print("forward 50%")
rover.forward(50)
time.sleep(2)
print('braking')
rover.brake()

print("forward 60%")
rover.forward(60)
time.sleep(2)
print('braking')
rover.brake()

print("forward 70%")
rover.forward(70)
time.sleep(2)
print('braking')
rover.brake()

print("reverse 0%")
rover.reverse(0)
time.sleep(2)
print('braking')
rover.brake()

print("reverse 10%")
rover.reverse(10)
time.sleep(2)
print('braking')
rover.brake()

print("reverse 20%")
rover.reverse(20)
time.sleep(2)
print('braking')
rover.brake()

print("reverse 30%")
rover.reverse(30)
time.sleep(2)
print('braking')
rover.brake()

print("reverse 40%")
rover.reverse(40)
time.sleep(2)
print('braking')
rover.brake()

print("reverse 50%")
rover.reverse(50)
time.sleep(2)
print('braking')
rover.brake()

print("reverse 60%")
rover.reverse(60)
time.sleep(2)
print('braking')
rover.brake()

print("reverse 70%")
rover.reverse(70)
time.sleep(2)
print('braking')
rover.brake()