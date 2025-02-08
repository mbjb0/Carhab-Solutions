# THIS MUST BE AT THE VERY TOP OF THE FILE
import os
os.environ['BLINKA_FT232H'] = '1'

import time
import board
import busio
from adafruit_pca9685 import PCA9685

# Constants
MAX_THROTTLE = 80
MIN_THROTTLE = 50
SERVO_CHANNEL = 0
THROTTLE_CHANNEL = 1

# Servo configuration
MAX_TURN_ANGLE = 60  # Maximum turn angle
SERVO_CENTER = 307   # Center position (1.5ms pulse)
SERVO_MIN = 205      # Full left position (1ms pulse)
SERVO_MAX = 409      # Full right position (2ms pulse)

# ESC configuration
ESC_NEUTRAL = 307    # Neutral position
ESC_MAX_FORWARD = 389  # Maximum forward position
ESC_MAX_REVERSE = 225  # Maximum reverse position

class Controls:
    def __init__(self):
        print("Initializing I2C")
        try:
            self.i2c = busio.I2C(board.SCL, board.SDA)
            self.pca = PCA9685(self.i2c, address=0x40)
            self.pca.frequency = 50  # 50Hz for standard servos
            self.turn_center()
            self.brake()
            print("Done initializing")
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

    def _set_pwm(self, channel, on, off):
        """Set PWM with boundary checking"""
        # Convert 12-bit value (0-4095) to 16-bit value (0-65535)
        duty_cycle = int((off / 4095) * 65535)
        duty_cycle = max(0, min(65535, duty_cycle))
        self.pca.channels[channel].duty_cycle = duty_cycle

    def _angle_to_pwm(self, angle):
        """Convert angle (-MAX_TURN_ANGLE to +MAX_TURN_ANGLE) to PWM value"""
        # First, constrain the angle
        angle = max(-MAX_TURN_ANGLE, min(MAX_TURN_ANGLE, angle))
        
        # Instead of using a ratio, map the angle directly to PWM range
        total_pwm_range = SERVO_MAX - SERVO_MIN
        total_angle_range = MAX_TURN_ANGLE * 2  # Total range from -MAX to +MAX
        
        # Calculate PWM value linearly
        pwm_per_degree = total_pwm_range / total_angle_range
        offset = angle * pwm_per_degree
        
        # Center the PWM value
        pulse = SERVO_CENTER + int(offset)
        
        # Final boundary check
        pulse = max(SERVO_MIN, min(SERVO_MAX, pulse))
        
        return pulse

    def _speed_to_pwm(self, speed, reverse=False):
        """Convert speed (0-100) to PWM value for ESC"""
        speed = max(0, min(100, speed))
        
        if reverse:
            # For reverse, interpolate between neutral and max reverse
            speed_range = ESC_NEUTRAL - ESC_MAX_REVERSE
            pulse = ESC_NEUTRAL - int((speed / 100) * speed_range)
        else:
            # For forward, interpolate between neutral and max forward
            speed_range = ESC_MAX_FORWARD - ESC_NEUTRAL
            pulse = ESC_NEUTRAL + int((speed / 100) * speed_range)
            
        return pulse

    def forward(self, speed):
        """Move forward at specified speed"""
        speed = MAX_THROTTLE if (speed + MIN_THROTTLE) > MAX_THROTTLE else (speed + MIN_THROTTLE)
        print(f"Forward Speed: {speed}%")
        pwm_value = self._speed_to_pwm(speed)
        self._set_pwm(THROTTLE_CHANNEL, 0, pwm_value)

    def reverse(self, speed):
        """Move in reverse at specified speed"""
        # Apply the same throttle limits as forward
        speed = MAX_THROTTLE if (speed + MIN_THROTTLE) > MAX_THROTTLE else (speed + MIN_THROTTLE)
        print(f"Reverse Speed: {speed}%")
        pwm_value = self._speed_to_pwm(speed, reverse=True)
        self._set_pwm(THROTTLE_CHANNEL, 0, pwm_value)

    def brake(self):
        """Stop the vehicle"""
        self._set_pwm(THROTTLE_CHANNEL, 0, ESC_NEUTRAL)

    def turn_center(self):
        """Center the steering"""
        self._set_pwm(SERVO_CHANNEL, 0, SERVO_CENTER)

    def turn(self, angle):
        """Turn to specified angle - immediate response"""
        pwm = self._angle_to_pwm(angle)
        self._set_pwm(SERVO_CHANNEL, 0, pwm)

    def turn_right(self, angle):
        """Turn right by specified angle"""
        scaled_angle = min(angle, MAX_TURN_ANGLE)
        pwm = self._angle_to_pwm(scaled_angle)
        print(f"Turn Right - Input: {angle}°, Scaled: {scaled_angle}°, PWM: {pwm}")
        self.turn(scaled_angle)

    def turn_left(self, angle):
        """Turn left by specified angle"""
        scaled_angle = max(-angle, -MAX_TURN_ANGLE)
        pwm = self._angle_to_pwm(scaled_angle)
        print(f"Turn Left - Input: {angle}°, Scaled: {scaled_angle}°, PWM: {pwm}")
        self.turn(scaled_angle)

    def cleanup(self):
        """Clean up and reset servos"""
        self.brake()
        self.turn_center()
        time.sleep(0.1)
        
        for channel in range(16):
            self._set_pwm(channel, 0, 0)
        
        self.pca.deinit()

if __name__ == "__main__":
    try:
        controls = Controls()
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    finally:
        if 'controls' in locals():
            controls.cleanup()