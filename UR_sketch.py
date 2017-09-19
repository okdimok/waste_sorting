
# coding: utf-8

# __Universal Robot Control__

# In[288]:

import urx #The UR library. $ pip install urx
import time, pickle #standard pythonic things. #picle is used to save objects to disk
import numpy as np #numpy is for computing in python
import math3d as m3d #the library, used by urx to represent transformations


# In Jupyter Notebook, use `Shift+Enter` to execute cell.
# 
# Use `Tab` to get input autocompletion. Try typing `np.save` and than pressing `Tab`, for example.
# 
# `B` to create an empty cell below the selected one.
# 
# Use `Shift+Tab` to get some info on the function, when inside parantheses.

# In[290]:

#You need to share your connection to the robot, and enable DHCP on your PC
#Or you can set up different static IP addresses on the robot and on your PC
#Use ping <robot_ip> in your terminal to check the setup
robot = urx.Robot("10.42.0.162", use_rt=True)


# In[373]:

#We have a small vacuum pump, connected to the digital output #0 of the robot
#Guess what is this code doing
robot.set_digital_out(0, 1)
time.sleep(1)
robot.set_digital_out(0, 0)


# In[297]:

#ckeck Installation tab of the robot PolyScope, TCP stands for tool central point
robot.set_tcp((0, 0, 0.05, 0, 0, 0)) 


# In[294]:

#be carefull! This really moves the robot! Guess the units and the variables
#Shift+Tab to help you
robot.translate((0.05, 0, 0.), acc=0.05, vel=0.05) #acceleration, velocity


# In[42]:

#same effect
robot.translate_tool((0, 0, 0.03), acc=0.05, vel=0.05)


# In[43]:

robot.get_pose() #rotation matrix + position


# In[44]:

robot.get_pos() #just position


# In[45]:

robot.getl() #position + rotation vector (see https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation)


# In[390]:

#defining obvious functions
def save_current_pos(fname):
    p = robot.getl()
    np.savetxt(fname, np.array(p))
       

def move_to_pos(fname, *args, **kwargs):
    p = np.loadtxt(fname)
    robot.movel(p, *args, **kwargs)
    
def print_pos(fname):
    p = np.loadtxt(fname)
    print(p)
    


# In[25]:


# save_current_pos('take_photo')


# In[42]:

def move_x_y(x, y):
    p = np.array(robot.getl())
    p[0] = x
    p[1] = y
    robot.movel(p, vel=0.05, acc=0.05)


# In[ ]:

move_to_pos("in", vel=0.05, acc=0.05)


# In[ ]:

robot.getl()


# In[ ]:

robot.x += 0.01 #1 hour 10 minutes up to this point for me ;)


# In[ ]:

p = robot.get_pos()


# In[ ]:

p.array


# In[ ]:

m3d.Vector(p.array)


# In[349]:

#This is a magic code to display the force on the TCP online
import threading
from IPython.display import display
import ipywidgets as widgets
import time

fw = widgets.HTML(
    value='',
    placeholder='No data',
    description='Force:',
)

def observe_force(fw):
    while True:
        fw.value = "<br>".join(["{0:5} {1:7.3f}".format(*c) for c in 
                                zip("x y z rx ry rz".split(), robot.get_tcp_force())])

thread = threading.Thread(target=observe_force, args=(fw,))
display(fw)
thread.start()



# In[387]:

#also important function
def observe_force_mean(number=100):
    global force
    while True:
        force = robot.get_tcp_force()
        for i in range(number):
            force = np.vstack((force, robot.get_tcp_force()))
        force_mean = force.mean(axis=0)
        force_std = force.std(axis=0)
        print ("\n".join(["{0:5} {1:7.3f}".format(*c) for c in 
                                zip("x y z rx ry rz".split(), force_mean)]))
        print ("="*20)
            


# __Here starts CV__

# In[331]:

get_ipython().magic('matplotlib notebook')

import cv2 #see the comments below
import matplotlib.pyplot as plt
import signal, datetime, time 
import numpy as np
from numpy import linalg
#to check: cv2.getBuildInformation()
print (*filter(lambda s: "FFMPEG" in s, cv2.getBuildInformation().split("\n"))) 
print (*filter(lambda s: "V4L" in s, cv2.getBuildInformation().split("\n"))) #video 4 linux

#To install opencv with ffmpeg in conda
#https://github.com/conda-forge/opencv-feedstock/
#pip uninstall opencv
#pip uninstall opencv-python
#conda unistall opencv
#conda install conda=4.0.11
#conda config --add channels conda-forge
#conda install opencv


# In[303]:

def signal_handler(signal, frame):
    # KeyboardInterrupt detected, exiting
    global is_interrupted
    is_interrupted = True


# In[391]:

get_ipython().run_cell_magic('bash', '', '#v4l devices\nls -d -1 /dev/* | grep video')


# In[319]:

vc = cv2.VideoCapture("/dev/video0") #0 for the first webcam, 1 for the second. This is V4L


# In[340]:

def imshow(frame, from_color_space='bgr'): #show a picture from webcam
    plt.figure()
    if from_color_space == 'bgr':
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # makes the blues image look real colored
    elif from_color_space == 'hsv':
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)    # makes the hsv image look real colored
    else:
        rgb_frame=frame
    return plt.imshow(rgb_frame)

def online_view(): #always update a picture from webcam
    if vc.isOpened(): # try to get the first frame
        is_capturing, frame = vc.read()
        webcam_preview = imshow(frame)   
    else:
        is_capturing = False

    signal.signal(signal.SIGINT, signal_handler)
    is_interrupted = False
    while is_capturing:
        is_capturing, frame = vc.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # makes the blues image look real colored
        webcam_preview.set_data(frame)
        plt.draw()

        try:    # Avoids a NotImplementedError caused by `plt.pause`
            plt.pause(0.05)
        except Exception:
            pass
        if is_interrupted:
            vc.release()
            break
    
def get_frame(): #get a picture from webcam
    timeout = time.time() + 0.1
    while time.time() < timeout:
        if vc.isOpened(): # try to get the first frame
            is_capturing, frame = vc.read()
        else:
            raise (Exception("Unable to capture"))
    cv2.imwrite("/tmp/frame_{}.png".format(datetime.datetime.now()), frame)
    return frame



# In[392]:

imshow(get_frame())
#online_view()


# In[172]:

# online_view()


# __The blob detection__

# In[181]:

move_to_pos('take_photo', acc=0.05, vel=0.05)


# In[182]:

#this is all the computer vision magic! Use verbose=2 to understand it;)
def find_colored_cubes_centers_pixels(colorMin, colorMax, blur=40, verbose=0):
    # Read image
    # im = cv2.imread("detection/cubes_4.png")
    im = get_frame()
    cv2.imwrite('test.png', im)

    if verbose > 0: imshow(im)

    # Blur image to remove noise
    im = cv2.GaussianBlur(im, (5,5), blur)

    if verbose > 2: imshow(im)

    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    #greenMin = (30, 90, 120) #./range-detector.py -i test.png -f HSV
    #greenMax = (150, 255, 255)

    # Sets pixels to white if in purple range, else will be set to black
    mask = cv2.inRange(hsv, colorMin, colorMax)

    # Bitwise-AND of mask and purple only image - only used for display
    masked = cv2.bitwise_and(im, im, mask= mask)

    if verbose > 2: imshow(masked)

    # dilate makes the in range areas larger
    mask = cv2.erode(mask, None, iterations=5)

    # mask = cv2.dilate(mask, None, iterations=8)


    # Bitwise-AND of mask and purple only image - only used for display
    masked = cv2.bitwise_and(im, im, mask= mask)

    if verbose > 1: imshow(masked)
    pts = []

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    if verbose > 1: print (len(cnts))

    # only proceed if at least one contour was found
    for c in cnts:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
    #     c = max(cnts, key=cv2.contourArea)
        (circle_center, radius) = cv2.minEnclosingCircle(c)

        # only proceed if the radius meets a minimum size
        if radius > 25 and radius < 200:
            if verbose > 1: print (radius)

            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            if center[1] > 400:
                continue
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            circle_center_int = tuple(map(int, circle_center))
            cv2.circle(im, circle_center_int, int(radius),
                (0, 255, 255), 2)
            cv2.circle(im, center, 5, (0, 0, 255), -1)
            cv2.circle(im, circle_center_int, 3,
                (0, 255, 255), -1)

            # update the points queue
            pts.append(circle_center)

    if verbose > 0: imshow(im)
    return pts


# In[183]:

green_min = (30, 75, 186) #./range-detector.py -i test.png -f HSV
green_max = (88, 255, 255)
im = find_colored_cubes_centers_pixels(green_min, green_max, verbose=1)


# In[186]:

def find_green_cubes_centers_pixels(*args, **kwargs):
    #maybe you will need to tune these values
    green_min = (30, 75, 186) #./range-detector.py -i test.png -f HSV
    green_max = (88, 255, 255)#
    return find_colored_cubes_centers_pixels(green_min, green_max, *args, **kwargs)

def find_plain_cubes_centers_pixels(*args, **kwargs):
    plain_min = (11, 45, 233) #./range-detector.py -i test.png -f HSV
    plain_max = (176, 93, 255)
    return find_colored_cubes_centers_pixels(plain_min, plain_max, *args, **kwargs)


# In[271]:

find_green_cubes_centers_pixels(verbose=1)


# In[256]:

# save_current_pos('green')


# In[270]:

move_to_pos('take_photo', acc=0.05, vel=0.05)


# In[257]:

# save_current_pos('plain')


# In[393]:

class RobotCamera(object):
    def calibrate(self, verbose=0):
        self.calibrate_rotation_and_scale(verbose=verbose)
        self.calibrate_shift(verbose=verbose)
    
    def calibrate_rotation_and_scale(self, verbose=0):
        scale, angle = self.calibrate_once(verbose=verbose)
        
        self.scale, self.angle = scale, angle
        if verbose > 0: print ("scale: {}, angle: {}".format(scale, angle))
        c, s = np.cos(self.angle), np.sin(self.angle)
        self.rotation_matrix = np.array([[-c, s], [s, c]]) #to maintain same orientation
        return self.scale, self.angle
        
    def calibrate_once(self, delta_pos=(0.05, 0, 0), verbose=0):
        """Calibrates the robot-camera scale, and angle between coordinate systems.
        Only one green cube in sight is required.
        
        It takes a photo, detects a green cube, than moves the robot on delta_pos, takes a photo and detects a cube.
        """
        if delta_pos[2] != 0:
            raise NotImplemented
        delta_pos = np.asarray(delta_pos)
        move_to_pos("take_photo", vel=0.05, acc=0.05)
        pos1 = np.array(find_green_cubes_centers_pixels(verbose=verbose)[0])

        robot.translate(delta_pos, acc=0.05, vel=0.05) #acceleration, velocity
        pos2 = np.array(find_green_cubes_centers_pixels(verbose=verbose)[0])
        if verbose > 0: print ("Pos 1: {}\nPos 2: {}\n".format(pos1, pos2))

        delta_pos_pix = (pos2 - pos1)
        
        if verbose > 1: print ("delta_pos_pix: {}\ndelta_pos: {}".format(delta_pos_pix, delta_pos[:2]))

        distance_pixels = linalg.norm(delta_pos_pix)
        distance_space = linalg.norm(delta_pos[:1])

        scale = distance_space/distance_pixels

        complex_pix = delta_pos_pix[0] + 1j*delta_pos_pix[1]
        complex_space = delta_pos[0] + 1j*delta_pos[1]
        #If the vectors are close to 0, that might need to be fixed
        angles = np.angle((complex_pix, complex_space)) #to get phase of comlex numbers.
        return scale, angles[1]-angles[0]
    
    def calibrate_shift(self, verbose=0):
        """Also the translation has to be calibrated. You need to know the correspondence between two points"""
        move_to_pos("take_photo", vel=0.05, acc=0.05)
        pos_pix = np.array(find_green_cubes_centers_pixels(verbose=verbose)[0])
        robot.set_freedrive(1)
        input("Now move the robot head to the green cube. Press Enter to confirm")
        robot.set_freedrive(0)
        pos_space = robot.getl()[:2]
        if verbose > 1: print("pos_space: {}".format(pos_space))
        move_to_pos("take_photo", vel=0.05, acc=0.05)
        
        proposed_coordinates = self.scale*self.rotation_matrix.dot(pos_pix)
        if verbose > 1: print ("proposed_coordinates: {}".format(proposed_coordinates))
        self.shift = pos_space - proposed_coordinates
        if verbose > 0: print ("shift: {}".format(self.shift))
        return self.shift

        

    def get_real_coords(self, point):
        #The main function to transform between the photo pixel coordinates and real world
        return self.scale*self.rotation_matrix.dot(point) + self.shift
    
    def test(self):
        #to tst the calibration
        move_to_pos("take_photo", vel=0.05, acc=0.05)
        cube_pix = find_green_cubes_centers_pixels()[0]
        cube_space = self.get_real_coords(cube_pix)
        print ("cube_pix: {}\ncube_space: {}".format(cube_pix, cube_space))
        p = np.array(robot.getl())
        p[:2] = cube_space[:2]
        p[2] = 0.005
        robot.movel(p, vel=0.05, acc=0.05)


# In[394]:

def take_cube(x, y, color):
    move_to_pos("take_photo", vel=0.5, acc=0.2)
    p = np.array(robot.getl())
    p[0] = x
    p[1] = y
    p[2] = 0.015
    robot.set_digital_out(0, 1)
    robot.movel(p, vel=0.5, acc=0.2)
    p[2] = 0.0
    robot.movel(p, vel=0.05, acc=0.2)
    get_frame()
    p[2] = 0.15
    robot.movel(p, vel=0.2, acc=0.1)
    move_to_pos(color, vel=0.5, acc=0.2)
    robot.set_digital_out(0, 0)


def sort():
    while True:
        move_to_pos("take_photo", vel=0.5, acc=0.2)
        green = find_green_cubes_centers_pixels()
        plain = find_plain_cubes_centers_pixels()
        if len(green) == 0 and len(plain) == 0:
            time.sleep(1)
            break
            continue

        take_green = len(green) > len(plain)

        box = green[0] if take_green else plain[0]
        color = 'green' if take_green else 'plain'
        x, y = rc.get_real_coords(box)
        
        take_cube(x, y, color)



# In[283]:

rc = RobotCamera()
rc.calibrate(verbose=2)


# In[245]:

robot.getl()


# In[346]:

sort()


# In[348]:

imshow(get_frame())

