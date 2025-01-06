from Jetson_instruct import JetsonController
import time

class Instruction:
    def init__(self):
        self.car = JetsonController()
        self.center_threshold = 10 #acceptable pixel distance for centered sign
        self.turn_left = ''
        self.turn_right = ''
        self.stop = ''
        self.u_turn = ''
        self.last_ID = 0
        self.last_cls = 0
        self.centerflag = 0
        self.stopflag = 0
        self.depth_threshold = 2 #acceptable depth distance to execute instruction

    def get_distance_to_center(self, bbox, frame_width, frame_height):
        centroid_x = (bbox[0] + bbox[2]) / 2
        center_x = frame_width / 2
        return centroid_x - center_x

    def left_loop(self, debug):
        self.car.turn_left
        self.car.move_forward
    
    def right_loop(self, debug):
        self.car.turn_right
        self.car.move_forward

    def stop_loop(self, debug):
        self.car.stop_motors
        time.sleep(2)
    
    def u_turn_loop(self, debug):
        self.car.spin_in_place

    # interpet sign is broken into 2 parts, following the instruction, and centering the sign
    # Center flag determines which step is taken, and is determined by the sign being centered
    # initially, then afterwards wether it maintains the same class or sort ID. 
    # Center flag is also reset if the sign is not close enough and the car must move
    # towards the sign (to ensure it doesnt get off track while moving forward)
    # The instruction step calls the various instruction loop functions

    def interpret_sign(self, tracker, frame_width, frame_height, depth, debug):
        x1,y1,x2,y2,id,cls = tracker

        if(debug == 1):
            depth = 0
        
        bbox = [x1,y1,x2,y2]
        distance = self.get_distance_to_center(bbox, frame_width,frame_height)
        if(self.centerflag == 0 and abs(distance) > self.center_threshold):
            if(distance > self.center_threshold):
                self.car.turn_left
                return
            elif(distance < (-1 * self.center_threshold)):
                self.car.turn_right
                return
        
        self.centerflag = 1
        
        if(depth > self.depth_threshold):
            self.car.move_forward
            self.centerflag = 0
            return

        if (cls == self.turn_left):
            self.left_loop(debug)
        
        elif (cls == self.turn_right):
            self.right_loop(debug)
        
        elif (cls == self.stop):
            if(self.stopflag == 0):
                self.stop_loop(debug)
                self.stopflag = 1
            else:
                self.car.move_forward
        
        elif(cls == self.u_turn):
            self.u_turn_loop(debug)
        
        if (self.last_cls != cls and self.last_ID != id):
            self.centerflag = 0
            self.stopflag = 0

        self.last_cls = cls
        self.last_ID = id


