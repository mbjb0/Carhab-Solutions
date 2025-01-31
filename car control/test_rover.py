from controls import Controls
import time
import signal
import sys

def signal_handler(sig, frame):
    print('Exiting...')
    if 'rover' in globals():
        rover.cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

try:
    rover = Controls()
    time.sleep(1)  # Give time for initialization
    
    # Test sequence
    print("Testing steering...")
    rover.turn_center()
    time.sleep(1)
    
    print("Testing gentle left turn...")
    rover.turn_left(30)
    time.sleep(1)
    
    print("Back to center...")
    rover.turn_center()
    time.sleep(1)
    
    print("Testing gentle right turn...")
    rover.turn_right(30)
    time.sleep(1)
    
    print("Back to center...")
    rover.turn_center()
    time.sleep(1)
    
    print("Testing throttle...")
    print("Small forward movement...")
    rover.forward(30)  # Very gentle forward movement
    time.sleep(2)
    
    print("Stopping...")
    rover.brake()
    time.sleep(1)
    
    rover.cleanup()
    
except Exception as e:
    print(f"Error: {e}")
    if 'rover' in globals():
        rover.cleanup()
    raise