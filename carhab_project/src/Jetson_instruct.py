"""
Test Script for RC Car Controls
This script provides a comprehensive test of all car movement functions.
It runs through a sequence of movements to verify proper operation of:
- Forward/backward movement
- Left/right turning
- Spinning in place
- Emergency stop and cleanup
"""

from controls import Controls
import time

def test_basic_movements():
    """
    Run through a complete test sequence of all car movements.
    Includes safety measures with try/finally to ensure proper cleanup.
    """
    # Initialize the control system
    rover = Controls()
    
    try:
        # TEST 1: Forward and Backward Movement
        print("\n=== Testing Forward/Backward Movement ===")
        
        # Forward movement test
        print("Testing forward movement...")
        rover.move_forward(speed=30)  # 30% speed
        time.sleep(2)  # Run for 2 seconds
        rover.stop_motors()
        time.sleep(1)  # Pause between movements
        
        # Backward movement test
        print("Testing backward movement...")
        rover.move_backward(speed=30)  # 30% speed
        time.sleep(2)
        rover.stop_motors()
        time.sleep(1)

        # TEST 2: Turning Capabilities
        print("\n=== Testing Turning Capabilities ===")
        
        # Left turn test
        print("Testing left turn...")
        rover.turn_left(angle=45)  # 45 degree turn
        time.sleep(1)
        rover.turn_center()  # Return to center
        time.sleep(1)

        # Right turn test
        print("Testing right turn...")
        rover.turn_right(angle=45)  # 45 degree turn
        time.sleep(1)
        rover.turn_center()  # Return to center
        time.sleep(1)

        # TEST 3: Spinning Maneuvers
        print("\n=== Testing Spin Maneuvers ===")
        
        # Test left spin
        print("Testing left spin...")
        rover.spin_in_place(direction="left", speed=30, duration=2.0)
        time.sleep(1)
        
        # Test right spin
        print("Testing right spin...")
        rover.spin_in_place(direction="right", speed=30, duration=2.0)

    finally:
        # Ensure cleanup happens even if tests fail
        print("\n=== Cleaning Up ===")
        rover.cleanup()

if __name__ == "__main__":
    # Only run tests if script is run directly
    print("Starting RC Car Movement Tests...")
    test_basic_movements()
    print("All tests completed.")