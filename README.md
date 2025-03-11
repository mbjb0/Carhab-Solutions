<img src="metallogo.png" alt="Carhab Solutions Logo" width="1000" height="350">
<p><strong>Carhab Solutions: Sign-Based Autonomous Navigation</strong></p>
<p>Carhab Solutions presents an innovative approach to autonomous vehicle navigation that relies exclusively on traffic sign recognition and depth sensing. Unlike traditional autonomous vehicles that depend on lane following algorithms, our system implements a point-to-point navigation strategy that's more adaptable to diverse environments.</p>
<p>The vehicle approaches the closest identified sign and executes its associated instruction, then locates the next sign in sequence. This modular solution provides greater flexibility for small-scale autonomous vehicles across various settings, from indoor delivery systems to inventory management.</p>
<br>
<p>Team Members: Mohammad Alburaidi (malburaidi@ucdavis.edu), Logan Field (lrfield@ucdavis.edu), Nasih Al-Barwani (nmalbarwani@ucdavis.edu), Pranav Nallapaneni</p>
<br>
<p><strong>Features</strong></p>
<p><strong>Sign-Based Navigation System</strong><br>
<img src="githubcompresseddemo.gif" alt="Demo gif" width="1000" height="350">
Our vehicle recognizes six types of traffic signs (left, right, forward, stop, caution, and u-turn) and executes corresponding maneuvers. The system treats signs as either destination indicators (left, right, u-turn) or behavior modifiers (caution, forward, stop) that affect the vehicle's approach behavior.</p>
<p><strong>Real-time Depth Sensing</strong><br>
Using an Intel RealSense camera, the vehicle accurately perceives distances to signs and obstacles, enabling precise positioning and collision avoidance during navigation.</p>
<p><strong>Finite State Machine Implementation</strong><br>
The driving algorithm is structured as a finite state machine that processes tracking input, state timing, and depth information to determine the optimal navigation path, allowing for complex behaviors like waiting at stop signs or slowing for caution signs.</p>
<p><strong>Adaptive Speed Control</strong><br>
The vehicle dynamically adjusts throttle based on the distance to target signs, slowing as it approaches them to ensure accurate execution of turning instructions and reduce collision risk.</p>
<br>
<p><strong>Technology Stack</strong></p>
<p>
- <strong>Vehicle Base:</strong> Modified Traxxas RC vehicle with improved bumper and bypass suspension<br>
- <strong>Vision System:</strong> Intel RealSense camera for RGB imaging and depth sensing<br>
- <strong>Object Detection:</strong> YOLOv5s model customized for traffic sign detection<br>
- <strong>Control Interface:</strong> FT232H Adafruit breakout board for motor control signals<br>
- <strong>Processing Platform:</strong> Laptop-based computing (with documented steps for Jetson TX2 implementation)
</p>
<br>
<p><strong>Getting Started</strong></p>
<p><strong>Hardware Requirements</strong>
<br>
- Traxxas RC vehicle<br>
- Intel RealSense camera (D435 or similar)<br>
- FT232H Adafruit breakout board<br>
- PCA9685 Adafruit PWM module<br>
- Laptop with USB 3.0 port
</p>
<p><strong>Software Installation</strong></p>
<p>1. Clone the repository</p>
<pre>
git clone https://github.com/yourusername/carhab-solutions.git
cd carhab-solutions
</pre>
<p>2. Install dependencies</p>
<pre>
pip install -r requirements.txt
</pre>
<p>3. Install PyTorch and YOLOv5</p>
<pre>
pip install torch torchvision
pip install ultralytics
</pre>
<p>4. Install Intel RealSense SDK</p>
<pre>
pip install pyrealsense2
</pre>
<br>
<p><strong>Sign System</strong></p>
<p>Our solution uses a dual-category sign system:</p>
<p><strong>Destination Signs</strong><br>
- <strong>Left Arrow:</strong> Vehicle turns left at sign<br>
- <strong>Right Arrow:</strong> Vehicle turns right at sign<br>
- <strong>U-Turn:</strong> Vehicle performs a 180° turn</p>
<p><strong>Modifier Signs</strong><br>
- <strong>Forward:</strong> Increases approach speed<br>
- <strong>Caution:</strong> Decreases approach speed<br>
- <strong>Stop:</strong> Pauses briefly before continuing</p>
<br>
<p><strong>Project Development</strong></p>
<p><strong>Model Training</strong><br>
The YOLOv5s model was trained on a custom dataset of printed traffic signs, photographed from multiple angles and distances. Transfer learning was applied after initial training on the German Traffic Sign Recognition Benchmark (GTSRB) dataset.</p>
<p>The custom dataset includes:<br>
- 739 original images (70-30 train-test split)<br>
- 1773 total images after augmentation<br>
- Significantly improved detection distance from ~1.2m to ~4.4m</p>
<p><strong>Driving Algorithm</strong><br>
The driving algorithm functions as a state machine with the following primary states:<br>
1. <strong>IDLE:</strong> Waiting for sign detection<br>
2. <strong>ALIGN:</strong> Centering on detected sign<br>
3. <strong>APPROACH:</strong> Moving toward the sign<br>
4. <strong>EXECUTE:</strong> Performing the sign's instruction<br>
5. <strong>POLLING:</strong> Searching for the next sign<br>
6. <strong>OBSTACLE:</strong> Reversing when obstacles are detected</p>
<br>
<p><strong>Future Work</strong></p>
<p>
- Integration with Jetson platforms for edge computing capabilities<br>
- Expansion of sign library to include more complex instructions<br>
- Implementation of environmental mapping for improved navigation<br>
- Development of multi-sign sequence interpretation for complex routing
</p>
<br>
<p><strong>Acknowledgements</strong></p>
<p>We extend our gratitude to Professor Chuah and Kartik for their help and guidance throughout the quarter.</p>
<br>
<p><strong>References</strong></p>
<p>
- <a href="https://github.com/NVIDIA-AI-IOT/jetracer">NVIDIA-AI-IOT/jetracer</a><br>
- <a href="https://github.com/noshluk2/ROS2-Autonomous-Driving-and-Navigation-SLAM-with-TurtleBot3">ROS2 Autonomous Driving and Navigation SLAM</a><br>
- <a href="https://github.com/Just-Jacksone/ARMS/tree/main">A.R.M.S. Project</a><br>
- <a href="https://github.com/ultralytics/yolov5">YOLOv5 Object Detection</a><br>
- <a href="https://medium.com/@jithin8mathew/estimating-depth-for-yolov5-object-detection-bounding-boxes-using-intel-realsense-depth-camera-a0be955e579a">Intel RealSense Depth Estimation</a>
</p>
