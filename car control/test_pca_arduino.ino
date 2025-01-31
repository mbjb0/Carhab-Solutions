#define MIN_THROTTLE 217
** #define MAX_THROTTLE 410**

Adafruit_PWMServoDriver servos = Adafruit_PWMServoDriver(0x40);

void initPCA() {
** servos.begin(); **
** servos.setPWMFreq(50); //Frecuecia PWM de 60Hz o T=16,66ms**
}

Then I init the motors
void initMotors() {
** servos.setPWM(2,0,204);**
** servos.setPWM(11,0,204);**
** delay(1000);**
}
servos.setPWM(2,0,poweryneg);
servos.setPWM(11,0,powerypos);
