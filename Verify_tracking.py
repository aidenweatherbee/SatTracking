# ------ import all necessary libraries ------

import os
import sys
import time
from typing import Sequence
#import rospy 
#from std_msgs.msg import Float64
import csv

import cv2
import fire
import numpy as np
from keras.models import model_from_json
from motpy import Detection, MultiObjectTracker, NpImage
from motpy.core import setup_logger
from motpy.detector import BaseObjectDetector
from motpy.testing_viz import draw_detection, draw_track
from pathlib import Path
import pyrealsense2 as rs
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# ------ all folders below must be in the same folder as this file ------
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device
import collections
import copy
import matplotlib
matplotlib.use('TkAgg')  # ---- specify backend ---- 
import matplotlib.pyplot as plt

x_saved = []
y_saved = []
z_saved = []
t_saved = []

logger = setup_logger(__name__, 'DEBUG', is_main=True)
# ------ YOLOV5 source code, inference task is done in this class ------

class ObjectDetector(BaseObjectDetector):

    def __init__(self):
        self.weights = ROOT / 'cubesat.pt'  # model.pt path(s)
        self.data = ROOT / 'cubesat.yaml'  # dataset.yaml path
        self.imgsz = (416, 224)  # inference size (height, width)
        self.conf_thres = 0.30  # confidence threshold
        self.iou_thres = 0.65  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.augment = False  # augmented inference
        self.half = False  # use FP16 half-precision inference
        self.dnn = False  # use OpenCV DNN for ONNX inference

        # Load model
        device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=device, dnn=self.dnn, data=self.data, fp16=self.half)
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size
        print('Yolo Model loaded successfully')

    # ------ Functions to get predictions in tensor format, and return boundary box coordinates, prediction scores, and class ID ------
    def _predict(self, image):

        pred = self.model(image, augment=self.augment, visualize=False)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        for i, det in enumerate(pred):
            det_ = det.to('cpu').detach().numpy()

            boxes = det_[:, :4]
            scores = det_[:,4]
            class_ids = det_[:,5]

        return boxes, scores, class_ids

    def process_image(self, image: NpImage) -> Sequence[Detection]:
        t0 = time.time()
        boxes, scores, class_ids = self._predict(image)
        elapsed = (time.time() - t0) * 1000.
        logger.debug(f'inference time: {elapsed:.3f} ms')
        return [Detection(box=b, score=s, class_id=l) for b, s, l in zip(boxes, scores, class_ids)]

# QuaternionEstimator class
# This class is used to estimate the quaternion orientation of the object that is detected by YOLOv5 given the bounding box coordinates
# the class must crop the images to the bounding box, resize the image to 120 by 120, and greyscale the images before passing them to the CNN
# the class uses a CNN model to estimate the quaternion orientation of the object
# the input to the crop function is the image, and the bounding box coordinates of the objects
# the output of the crop function is the cropped and resized and greyscale image
# we want to convert the quaternion to euler angles. The quaternion is a 4 dimensional vector, and the euler angles are 3 dimensional vectors

# input the necessary libraries to convert the quaternions to euler angles
from scipy.spatial.transform import Rotation as R
import math
class QuaternionEstimator:
    def __init__(self):
        # load json and create model
        json_file = open('C:/Users/Aiden/Desktop/dummy-sat/MDA-Aiden/Cone1_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        self.loaded_model.load_weights("C:/Users/Aiden/Desktop/dummy-sat/MDA-Aiden/Cone1_model.h5")
        print("Loaded model from disk")

    def crop(self, img, x1, y1, x2, y2):
        # ------ crop the image to the bounding box ------
        img = img[y1:y2, x1:x2]
        try:
            # ------ resize the image to 64 by 64 ------
            img = cv2.resize(img, (64, 64))
            # ------ convert the image to greyscale ------
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=3)
            pred = self.loaded_model.predict(img)
            
            #convert the quaternion to euler angles
            r = R.from_quat(pred)
            euler = r.as_euler('xyz', degrees=True)
            return euler

            
        except cv2.error as e:
            print('Invalid frame')
            return None



#------ Run object tracking task ------
def run(video_downscale: float = 1.,
        architecture: str = 'ssdlite320',
        confidence_threshold: float = 0.3,
        tracker_min_iou: float = 0.65,
        show_detections: bool = False,
        track_text_verbose: int = 0,
        device: str = 'cuda',
        viz_wait_ms: int = 1):

    #rospy.init_node('Ros_tracking', anonymous=True)
    #pub = rospy.Publisher('tracking', Float64, queue_size=10)
    draw_trajectory = True
    draw_trajectory_3D = True
    cmap = plt.get_cmap("tab10")

    detector = ObjectDetector()
    #cap_fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap_fps = 10
    tracker = MultiObjectTracker(
        dt=1 / cap_fps,
        tracker_kwargs={'max_staleness': 5},
        model_spec={'order_pos': 1, 'dim_pos': 2,
                    'order_size': 0, 'dim_size': 2,
                    'q_var_pos': 5000., 'r_var_pos': 0.1},
        matching_fn_kwargs={'min_iou': tracker_min_iou,
                            'multi_match_min_iou': 0.93})

    trajectory = []
    existing_trajectories = []

    #------ Realsense camera configuration, pipeline, and 

    config = rs.config()
    config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30) # set realsense color configuration
    config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30) # set realsense depth configuration

    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    # align depth and color streams
    align_to = rs.stream.color
    align = rs.align(align_to)

    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    
    estimator = QuaternionEstimator()
    # ------ start of object tracking using IOU ------
    # add timestamps to all detections and save them to a list
    # each timestamped detection will be compared to the known locations of the object to determine the error in the prediction of the object location
    # the error will be used to determine the confidence of the object location

    while True: #not rospy.is_shutdown():

        if(len(trajectory) > 0):
            existing_trajectories = [x for x in range(len(trajectory))]

        frames = pipeline.wait_for_frames()
        is_gone =True # tracker for executing to ros

        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame or not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        im = img[8:232,4:420] # image is cropped to fit YOLO   

        im0 = img.copy()
        im = im[np.newaxis, :, :, :]        

        # Stack
        im = np.stack(im, 0)

        # Convert
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        im = np.ascontiguousarray(im)

        im = torch.from_numpy(im).to(device)
        im = im.half() if detector.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # detect objects in the frame
        detections = detector.process_image(im)

        # track detected objects
        _ = tracker.step(detections=detections)
        active_tracks = tracker.active_tracks(min_steps_alive=3)
        
        # if it is the first prediction, we call this the start time 
        if len(t_saved) == 0:
            start_time = time.time()
        # take the timestamp of the current frame in seconds. the timestamp should begin from 0 at the first frame

        timestamp = time.time() - start_time

        # visualize and show detections and tracks
        if show_detections:
            for det in detections:
                draw_detection(im0, det)

        # ------ draw the trajectories of all tracks ------

        for track in active_tracks:
            draw_track(im0, track, thickness = 3)
            new_object = True

            # calculate center point from bounding box
            x = int ((track[1][0] + track[1][2])/2) 
            y = int ((track[1][1] + track[1][3])/2)

            if(x > 420):
                x = 420
            elif(x < 0):
                x = 0

            if(y > 232):
                y = 232
            if(y < 0):
                y = 0

            try:
                dist = depth_frame.get_distance(x , y + 16)*1000 # distance from camera to the center point of the image
            except:
                dist = depth_frame.get_distance(x , y)*1000 # distance from camera to the center point of the image near edge

            # calculation of trajectory, if no trajectories, a new one is added
            # each trajectories name and coordinates is stored as an array
            if(len(trajectory)==0):
                xy_traj = collections.deque([],20)
                xyz_traj = collections.deque([],20)
                xy_traj.append((x, y))
                xyz_traj.append((dist*(x - intr.ppx)/intr.fx - 35, dist*(y+16 - intr.ppy)/intr.fy, dist))
                trajectory.append([track[0], xy_traj, xyz_traj])                
            else: #trajectories updates with new coordinates
                is_gone = False
                for i in range(len(trajectory)):
                    if(track[0] == trajectory[i][0]):
                        trajectory[i][1].append((x, y))
                        trajectory[i][2].append((dist*(x - intr.ppx)/intr.fx - 35, dist*(y+16 - intr.ppy)/intr.fy, dist))
                        new_object = False
                        existing_trajectories.remove(i)

                if(new_object): #if there is a trajectorie with a new id, we add it to the array
                    xy_traj = collections.deque([],20)
                    xyz_traj = collections.deque([],20)
                    xy_traj.append((x, y))
                    xyz_traj.append((dist*(x - intr.ppx)/intr.fx - 35, dist*(y+16 - intr.ppy)/intr.fy, dist))
                    trajectory.append([track[0], xy_traj, xyz_traj])                 


        if(len(existing_trajectories) > 0):
            for k in range(len(existing_trajectories)):
                if (len(trajectory) > 1):
                    trajectory.pop(existing_trajectories[k])

        if(len(trajectory) > 0):
            if is_gone == False:
                m = len(trajectory[0][2]) -1
                tmpx = np.array(copy.deepcopy(trajectory[0][2][m][0]))
                tmpy = np.array(copy.deepcopy(trajectory[0][2][m][1]))
                tmpz = np.array(copy.deepcopy(trajectory[0][2][m][2]))
                # print the x, y, z coordinates of the object and the time it was detected
                print("x: ", tmpx, "y: ", tmpy, "z: ", tmpz, "time: ", timestamp)

                # save the x, y, z coordinates of the object and the time it was detected so that it can be saved to a csv file later
                x_saved.append(tmpx)
                y_saved.append(tmpy)
                z_saved.append(tmpz)
                t_saved.append(timestamp)




                # use the functions in QuaternionEstimator to estimate the quaternions
                # the inputs are the image and the bounding box coordinates
                # the outputs are the quaternions
                # use the crop function in QuaternionEstimator to crop the image to the bounding box for input to the model
                # use the predict function in QuaternionEstimator to predict the quaternions
                # the predict function takes the cropped image from the crop function as input and outputs the quaternions


                cropped_image = estimator.crop(im0, int(track[1][0]), int(track[1][1]), int(track[1][2]), int(track[1][3]))
                print('Euler:', cropped_image)

                #QuaternionEstimator.crop(im0, int(track[1][0]), int(track[1][1]), int(track[1][2]), int(track[1][3]))
                #quat = QuaternionEstimator.predict()
                #print the quaternions and the coordinates to the topic
                #print('quaternions: ',quat)
                #pub.publish(tmpx,tmpy,tmpz,quat[0],quat[1],quat[2],quat[3])

        # show the image
        cv2.imshow('frame', im0)
        c = cv2.waitKey(viz_wait_ms)
        if c == ord('q'):
            break

    #import the package needed to save the x, y, z coordiantes to a csv file
    
    # save the x, y, z coordinates of the object and the time it was detected to a csv file
    with open('C:/Users/Aiden/Desktop/dummy-sat/MDA-Aiden/coordinates.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "z", "time"])
        for i in range(len(t_saved)):
            writer.writerow([x_saved[i], y_saved[i], z_saved[i], t_saved[i]])




    # also plot the points in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_saved, y_saved, z_saved, c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

if __name__ == '__main__':
        fire.Fire(run)
