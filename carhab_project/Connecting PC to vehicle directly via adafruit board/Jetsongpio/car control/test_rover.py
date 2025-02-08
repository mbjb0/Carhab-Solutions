# THIS MUST BE AT THE VERY TOP OF THE FILE
import os
os.environ['BLINKA_FT232H'] = '1'

from controls import Controls
import time

def main():
    try:
        controls = Controls()
        
        # Test sequence
        print("Starting test sequence...")
        
        # Add your test code here
        # For example:
        controls.turn_center()
        time.sleep(1)
        controls.forward(30)
        time.sleep(2)
        controls.brake()
        
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'controls' in locals():
            controls.cleanup()

if __name__ == "__main__":
    main()