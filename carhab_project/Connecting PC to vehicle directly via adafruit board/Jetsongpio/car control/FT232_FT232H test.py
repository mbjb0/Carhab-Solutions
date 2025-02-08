import os
os.environ['BLINKA_FT232H'] = '1'

import board
import busio

# Try to initialize I2C
try:
    i2c = busio.I2C(board.SCL, board.SDA)
    print("FT232H I2C initialization successful!")
    
    # Scan for I2C devices
    print("\nScanning for I2C devices...")
    while not i2c.try_lock():
        pass
    
    try:
        devices = i2c.scan()
        if devices:
            print("Found I2C devices at addresses:", [hex(x) for x in devices])
            if 0x40 in devices:
                print("PCA9685 found at address 0x40!")
            else:
                print("PCA9685 not found at address 0x40")
        else:
            print("No I2C devices found")
    finally:
        i2c.unlock()
        
except Exception as e:
    print(f"Error: {e}")
    print("\nTroubleshooting tips:")
    print("1. Make sure FT232H is properly connected")
    print("2. Verify driver installation in Device Manager")
    print("3. Check your wiring connections:")
    print("   - FT232H SCK/SCL → PCA9685 SCL")
    print("   - FT232H D1/SDA → PCA9685 SDA")
    print("   - FT232H GND → PCA9685 GND")
    print("   - FT232H 3V → PCA9685 VCC")
    print("4. Try unplugging and replugging the FT232H")