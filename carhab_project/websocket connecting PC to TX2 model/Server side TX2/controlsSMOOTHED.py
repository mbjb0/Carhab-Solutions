import time
import smbus
try:
    import Jetson.GPIO as GPIO
except ImportError:
    print("Installing Jetson.GPIO...")
    import subprocess
    subprocess.check_call(["sudo", "pip3", "install", "Jetson.GPIO"])
    import Jetson.GPIO as GPIO

# Constants
MAX_THROTTLE = 80
MIN_THROTTLE = 50
SERVO_CHANNEL = 0
THROTTLE_CHANNEL = 1

class Controls:
    def __init__(self):
        print("Initializing GPIO")
        try:
            GPIO.cleanup()
        except:
            pass
        
        try:
            GPIO.setmode(GPIO.BOARD)
        except ValueError:
            current_mode = GPIO.getmode()
            print(f"GPIO mode was already set to: {current_mode}")
            if current_mode != GPIO.BOARD:
                GPIO.cleanup()
                GPIO.setmode(GPIO.BOARD)
        
        print("Initializing I2C")
        self.i2c = smbus.SMBus(1)
        self.pca_address = 0x40
        
        # Initialize PCA9685
        self._write_byte(0x00, 0x10)  # Sleep mode
        time.sleep(0.025)
        self._write_byte(0xFE, 0x79)  # Set PWM frequency to 50Hz
        time.sleep(0.025)
        self._write_byte(0x00, 0x00)  # Normal mode
        time.sleep(0.025)
        
        # Center the servo and stop the motor
        self.turn_center()
        self.brake()
        
        print("Done initializing")

    def _write_byte(self, reg, value):
        """Write a byte to the I2C device"""
        self.i2c.write_byte_data(self.pca_address, reg, value)

    def _get_current_pwm(self, channel):
        """Get current PWM value for channel"""
        # Read the current "off" time
        off_l = self.i2c.read_byte_data(self.pca_address, 0x08 + 4 * channel)
        off_h = self.i2c.read_byte_data(self.pca_address, 0x09 + 4 * channel)
        return (off_h << 8) + off_l

    def _set_pwm(self, channel, on, off):
        """Set PWM with boundary checking"""
        # Ensure PWM values are within valid range (0-4095)
        off = max(0, min(4095, off))
        on = max(0, min(4095, on))
        
        self._write_byte(0x06 + 4 * channel, on & 0xFF)
        self._write_byte(0x07 + 4 * channel, on >> 8)
        self._write_byte(0x08 + 4 * channel, off & 0xFF)
        self._write_byte(0x09 + 4 * channel, off >> 8)

    def _angle_to_pwm(self, angle):
        """Convert angle (0-180) to PWM value for servo"""
        # For most RC servos:
        # 1ms pulse (value 205) = -90 degrees
        # 1.5ms pulse (value 307) = 0 degrees
        # 2ms pulse (value 409) = +90 degrees
        angle = max(0, min(180, angle))  # Constrain angle
        pulse = int(205 + ((angle / 180) * (409 - 205)))
        return pulse

    def _speed_to_pwm(self, speed):
        """Convert speed (0-100) to PWM value for ESC"""
        # For most RC ESCs:
        # 1.1ms pulse (value 225) = full reverse
        # 1.5ms pulse (value 307) = neutral
        # 1.9ms pulse (value 389) = full forward
        speed = max(0, min(100, speed))  # Constrain speed
        neutral = 307
        max_forward = 389
        speed_range = max_forward - neutral
        pulse = neutral + int((speed / 100) * speed_range)
        return pulse

    def smooth_turn(self, target_angle, steps=10, delay=0.02):
        """Smoothly turn to target angle"""
        current_pwm = self._get_current_pwm(SERVO_CHANNEL)
        target_pwm = self._angle_to_pwm(target_angle)
        step_size = (target_pwm - current_pwm) / steps
        
        for i in range(steps):
            next_pwm = int(current_pwm + step_size * (i + 1))
            self._set_pwm(SERVO_CHANNEL, 0, next_pwm)
            time.sleep(delay)

    def forward(self, speed):
        """Move forward at specified speed"""
        speed = MAX_THROTTLE if (speed + MIN_THROTTLE) > MAX_THROTTLE else (speed + MIN_THROTTLE)
        print(f"Speeding up to {speed}%")
        pwm_value = self._speed_to_pwm(speed)
        self._set_pwm(THROTTLE_CHANNEL, 0, pwm_value)

    def brake(self):
        """Stop the vehicle"""
        print("Braking")
        # Set to neutral position
        self._set_pwm(THROTTLE_CHANNEL, 0, 307)

    def turn_center(self):
        """Center the steering"""
        print("Centered")
        self._set_pwm(SERVO_CHANNEL, 0, 307)  # 307 is center position (1.5ms pulse)

    def turn(self, angle):
        """Turn to specified angle with smoothing"""
        # Constrain angle between 0 and 180
        angle = max(0, min(180, angle))
        self.smooth_turn(angle)

    def turn_right(self, angle):
        """Turn right by specified angle"""
        print(f"Turning right {angle} degrees")
        self.smooth_turn(90 + angle)

    def turn_left(self, angle):
        """Turn left by specified angle"""
        print(f"Turning left {angle} degrees")
        self.smooth_turn(90 - angle)

    def cleanup(self):
        """Clean up GPIO and reset servos"""
        print("Cleaning up")
        # Set everything to neutral/center before shutting down
        self.brake()
        self.turn_center()
        time.sleep(0.1)  # Give time for servos to reach position
        
        # Turn off all channels
        for channel in range(16):
            self._set_pwm(channel, 0, 0)
            
        try:
            GPIO.cleanup()
        except:
            pass

# Usage example:
if __name__ == "__main__":
    try:
        controls = Controls()
        # Add your control sequence here
        
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    finally:
        if 'controls' in locals():
            controls.cleanup()