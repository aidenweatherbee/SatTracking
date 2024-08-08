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
#from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, strip_optimizer, xyxy2xywh)
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

intr = collections.namedtuple('intr', ['ppx', 'ppy', 'fx', 'fy'])
intr.ppx = 315.4064025878906
intr.ppy = 243.17507934570312
intr.fx = 385.762451171875
intr.fy = 385.762451171875

logger = setup_logger(__name__, 'DEBUG', is_main=True)
# ------ YOLOV5 source code, inference task is done in this class ------

def draw_detection1(img, det, class_names=None, show_label=True):
    c1, c2 = (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr))
    center = (int((det.xtl + det.xbr) / 2), int((det.ytl + det.ybr) / 2))  # calculate center of the bounding box
    cv2.rectangle(img, c1, c2, (255, 0, 0), 2)  # draw bounding box
    if show_label:
        cls = int(det.label)
        label = f'{class_names[cls]} {det.score:.2f}' if class_names else f'{cls} {det.score:.2f}'
        tf = max(0, min(center[1] - 5, img.shape[0] - 12))
        cv2.putText(img, label, (c1[0], tf), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(img, center, 6, (0, 0, 255), -1)  # draw dot at the center of the bounding box with a larger radius

        
class ObjectDetector(BaseObjectDetector):

    def __init__(self):
        self.weights = ROOT / 'C:/Users/Aiden/Desktop/dummy-sat/MDA-Aiden/yolov5/best5.pt'  # model.pt path(s)
        self.data = ROOT / 'C:/Users/Aiden/Desktop/dummy-sat/MDA-Aiden/yolov5/best1.yaml'  # dataset.yaml path
        self.imgsz = (640, 480)  # inference size (height, width)
        self.conf_thres = 0.20  # confidence threshold
        self.iou_thres = 0.25  # NMS IOU threshold
        self.max_det = 2  # maximum detections per image
        self.device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.classes = 1, 2  # filter by class: --class 0, or --class 0 2 3
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

# create a second class that will detect the center of the cone using the image cropped to the first detectors bounding box

class CenterDetector(BaseObjectDetector):
    def __init__(self):
        #self.weights = ROOT / 'C:/Users/Aiden/Desktop/dummy-sat/MDA-Aiden/yolov5/center-of-cone.v1i.yolov5pytorch/center_of_cone.pt'  # model.pt path(s)
        self.weights = ROOT / 'C:/Users/Aiden/Desktop/dummy-sat/MDA-Aiden/yolov5/center-of-cone.v1i.yolov5pytorch/best.pt'
        #self.data = ROOT / 'C:/Users/Aiden/Desktop/dummy-sat/MDA-Aiden/yolov5/center-of-cone.v1i.yolov5pytorch/center-of-cone.yaml'  # dataset.yaml path
        self.data = ROOT / 'C:/Users/Aiden/Desktop/dummy-sat/MDA-Aiden/yolov5/center-of-cone.v1i.yolov5pytorch/data.yaml'
        self.imgsz = (640, 640)  # inference size (height, width)
        self.conf_thres = 0.01  # confidence threshold
        self.iou_thres = 0.01  # NMS IOU threshold
        self.max_det = 1  # maximum detections per image
        self.device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.classes = 0  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.augment = False  # augmented inference
        self.half = False  # use FP16 half-precision inference
        self.dnn = False  # use OpenCV DNN for ONNX inference

        # Load model
        device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.model_center = DetectMultiBackend(self.weights, device=device, dnn=self.dnn, data=self.data, fp16=self.half)
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size
        print('Yolo Model loaded successfully')

    def _predict1(self, image):

        pred = self.model(image, augment=self.augment, visualize=False)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        for i, det in enumerate(pred):
            det_ = det.to('cpu').detach().numpy()

            boxes = det_[:, :4]
            scores = det_[:,4]
            class_ids = det_[:,5]

        return boxes, scores, class_ids

    def process_image1(self, image: NpImage) -> Sequence[Detection]:
        t0 = time.time()
        boxes, scores, class_ids = self._predict1(image)
        elapsed = (time.time() - t0) * 1000.
        logger.debug(f'inference time: {elapsed:.3f} ms')
        return [Detection(box=b, score=s, class_id=l) for b, s, l in zip(boxes, scores, class_ids)]



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
    center_detector = CenterDetector()

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

    # Set up the paths to the directories containing the images
    rgb_dir = "F:\\Test_position_Images1\\rgb"
    depth_dir = "F:\\Test_position_Images1\\depth"

    # Get a list of all the image files in the RGB directory
    rgb_files = os.listdir(rgb_dir)
    
    
    estimator = QuaternionEstimator()
    # ------ start of object tracking using IOU ------
    # add timestamps to all detections and save them to a list
    # each timestamped detection will be compared to the known locations of the object to determine the error in the prediction of the object location
    # the error will be used to determine the confidence of the object location

    for rgb_file in rgb_files:
        # Read the RGB image
        rgb_image_path = os.path.join(rgb_dir, rgb_file)
        color_frame = cv2.imread(rgb_image_path)
    
        # Get the corresponding depth image filename
        depth_filename = rgb_file
    
        # Construct the path to the depth image
        depth_image_path = os.path.join(depth_dir, depth_filename)
    
        # Read the depth image
        depth_frame = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)

        if(len(trajectory) > 0):
            existing_trajectories = [x for x in range(len(trajectory))]

    
        img = np.asanyarray(color_frame)
        
        # crop the image to fit yolo(640 x 480)
        im = img[0:480, 0:640]

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
            #start_time = time.time()
            # the name of the image contains a timestamp in hte format: img84time_1680293891.4585989. we want to extract the time for our timestamp. the time is the number after the word time_ in the name. use the name to extract the time
            
            # extract the whole number from the name by taking everything after the keywork time_
            time_string = rgb_file.split('time_')[1]
            # extract the number including all numbers after the decimal point
            time_string = time_string.split('.')[0] + '.' + time_string.split('.')[1]
            # convert the string to a float
            start_time = float(time_string)
            print('Start time: ', start_time)
            
        # take the timestamp of the current frame in seconds. the timestamp should begin from 0 at the first frame using the same extraction technique for the start time
        time_string = rgb_file.split('time_')[1]
        time_string = time_string.split('.')[0] + '.' + time_string.split('.')[1]
        timestamp = float(time_string) - start_time
        print('Current time: ', timestamp)

        # visualize and show detections and tracks
        #if show_detections:
        #    for det in detections:
        #        draw_detection1(im0, det)

        # ------ draw the trajectories of all tracks ------

        for track in active_tracks:
            # i want to add a second detection that is used on an image of the bounding box, so we need to crop the image to the bounding box
        
            # crop the image to the bounding box 
            crop_img_before = img[int(track[1][1]):int(track[1][3]), int(track[1][0]):int(track[1][2])]
            # crop the depth image too 
            crop_depth = depth_frame[int(track[1][1]):int(track[1][3]), int(track[1][0]):int(track[1][2])]
            
            # resize to 128 x 128 
            crop_img = cv2.resize(crop_img_before, (640, 640))
            cv2.imshow("cropped", crop_img)
            #crop_img = np.asanyarray(crop_img)
            
            
            crop_im = crop_img[np.newaxis, :, :, :]        

            # Stack
            crop_im = np.stack(crop_im, 0)

            # Convert
            crop_im = crop_im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            crop_im = np.ascontiguousarray(crop_im)

            crop_im = torch.from_numpy(crop_im).to(device)
            crop_im = crop_im.half() if detector.model.fp16 else crop_im.float()  # uint8 to fp16/32
            crop_im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(crop_im.shape) == 3:
                crop_im = crop_im[None]  # expand for batch dim
            # setup and use the class CenterDetector to detect the center of the cone from the cropped image
            center = center_detector.process_image1(crop_im)
            
            # print center
            print("center:", center)
            
            
            
            active_tracks2 = center
            
            x = 0
            y = 0
            
            # get the bounding box coordinates from activetracks2
            for track2 in active_tracks2:
                
                print("track2:", track2)
                print("track:", track)

                # the x y coordinates of the second detections bounding box
                x3_1 = track2.box[0]
                y3_1 = track2.box[1]
                x3_2 = track2.box[2]
                y3_2 = track2.box[3]

                
                print("crop_img_before shape:", crop_img_before.shape)
                print("crop_img shape:", crop_img.shape)
                
                print("x3_1:", x3_1)
                print("y3_1:", y3_1)
                print("x3_2:", x3_2)
                print("y3_2:", y3_2)
                


                
                # crop and display the image from the 128 by 128 image 
                crop_img2 = crop_img[int(y3_1):int(y3_2), int(x3_1):int(x3_2)]
                cv2.imshow("cropped2", crop_img2)
                

                # get the bounding box coordinates of the object in the original images from the first yolo detection
                x1 = int(track[1][0])
                y1 = int(track[1][1])
                x2 = int(track[1][2])
                y2 = int(track[1][3])
            


                # Calculate the ratio of the dimensions of the cropped image and the original image as follows:
                # width_ratio = original_image_width / cropped_image_width
                # height_ratio = original_image_height / cropped_image_height

                width_ratio = (x2 - x1) / 640
                height_ratio = (y2 - y1) / 640

                # Scale the bounding box coordinates from the 128 by 128 back to the size of crop_img_before
                x3_1 = x3_1 * width_ratio
                y3_1 = y3_1 * height_ratio
                x3_2 = x3_2 * width_ratio
                y3_2 = y3_2 * height_ratio
                
                # crop the cropped image again to the new bounding box coordinates and display the image
                #crop_img_before2 = crop_img_before[int(y3_1):int(y3_2), int(x3_1):int(x3_2)]
                #cv2.imshow("cropped2", crop_img_before2)
                #v2.waitKey(100)

                # Calculate the center point of the scaled bounding box
                x3_center = int((x3_1 + x3_2) / 2)
                y3_center = int((y3_1 + y3_2) / 2)

                # Convert the coordinates from being relative to the cropped image, to being relative to the original 640 x 480 image
                # This simply adds the pixels to the top left corner of the bounding box coordinates to get the true coordinates 
                x = x3_center + x1
                y = y3_center + y1
                
                # display the coordinates on hte original 640 by 480 image
                cv2.circle(im0, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow("im0", im0)
                


                 
                
                
            
            
            draw_track(im0, track, thickness = 3)
            new_object = True


            if(x > 640):
                x = 639
            elif(x < 0):
                x = 0

            if(y > 480):
                y = 479
            if(y < 0):
                y = 0

        
                
            dist = depth_frame[y,x]
            print(dist)
            # wait 
                

            # calculation of trajectory, if no trajectories, a new one is added
            # each trajectories name and coordinates is stored as an array
            if(len(trajectory)==0):
                xy_traj = collections.deque([],20)
                xyz_traj = collections.deque([],20)
                xy_traj.append((x, y))
                xyz_traj.append((dist*(x - intr.ppx)/intr.fx, dist*(y - intr.ppy)/intr.fy, dist))
                trajectory.append([track[0], xy_traj, xyz_traj])    
                is_gone = True            
            else: #trajectories updates with new coordinates
                is_gone = False
                for i in range(len(trajectory)):
                    if(track[0] == trajectory[i][0]):
                        trajectory[i][1].append((x, y))
                        trajectory[i][2].append((dist*(x - intr.ppx)/intr.fx, dist*(y - intr.ppy)/intr.fy, dist))
                        new_object = False
                        existing_trajectories.remove(i)

                if(new_object): #if there is a trajectorie with a new id, we add it to the array
                    xy_traj = collections.deque([],20)
                    xyz_traj = collections.deque([],20)
                    xy_traj.append((x, y))
                    xyz_traj.append((dist*(x - intr.ppx)/intr.fx, dist*(y - intr.ppy)/intr.fy, dist))
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
