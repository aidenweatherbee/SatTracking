import cv2
import numpy as np
from keras.models import model_from_json
from scipy.spatial.transform import Rotation as R
import os
import csv
from scipy.spatial.transform import Rotation as R
import json
from keras import backend as K
# this code will use the class QuaternionEstimator to estimate the roll pitch and yaw of an object from cropped images, and save them to a csv file with the image name.
flag = 0
img = []
quat = []
roll = []
pitch = []
yaw = []
true_roll = []
true_pitch = []
true_yaw = []
path = os.path.join('F:\pose\cropped3-first600')

# create another quaternion estimator class that only loads the model once and then uses it repeatedly
json_file = open('C:/Users/Aiden/Desktop/dummy-sat/MDA-Aiden/Sat5_model.json', 'r')
loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("C:/Users/Aiden/Desktop/dummy-sat/MDA-Aiden/Sat5_model.h5")
print("Loaded model from disk")


class QuaternionEstimator1: # this class loads the model once and then uses it repeatedly

    def predict1(self, img):
        IMG_SIZE = 128
        new_array = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        new_array = np.array(new_array).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        new_array = new_array/255
        pred = loaded_model.predict(new_array)
            
        #convert the quaternion to euler angles
        r = R.from_quat(pred)
        euler = r.as_euler('xyz', degrees=True)
        return euler


class QuaternionEstimator:
    def __init__(self):
        # load json and create model
        json_file = open('C:/Users/Aiden/Desktop/dummy-sat/MDA-Aiden/Sat5_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        self.loaded_model.load_weights("C:/Users/Aiden/Desktop/dummy-sat/MDA-Aiden/Sat5_model.h5")
        print("Loaded model from disk")

    def predict(self, img):
        IMG_SIZE = 128
        #new_array = smart_resize(img, (IMG_SIZE, IMG_SIZE), interpolation='bilinear')
        #img = img/255
        new_array = cv2.cvtColor( new_array, cv2.COLOR_BGR2GRAY)
        pred = self.loaded_model.predict(new_array)
            
        #convert the quaternion to euler angles
        r = R.from_quat(pred)
        euler = r.as_euler('xyz', degrees=True)
        return euler

# load the images from the folder 'F:\pose\cropped1-first600'


# create a csv file to save the estimated roll pitch and yaw and the true roll pitch and yaw
with open('C:/Users/Aiden/Desktop/dummy-sat/MDA-Aiden/estimated_angles6.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Estimated Roll', 'Estimated Pitch', 'Estimated Yaw', 'True Roll', 'True Pitch', 'True Yaw'])


for file in os.listdir(path):
    
    # if the file is a json file, get the true roll pitch and yaw from the json file
    if file.endswith(".json"):
        
        # get the quaternion from the json file under 'quaternion_xyzw' and convert it to euler angles
        with open('F:/pose/cropped3-first600/' + file) as f:
            data = json.load(f)
            quat = data['quaternion_xyzw']
            r = R.from_quat(quat)
            euler1 = r.as_euler('xyz', degrees=True)
            true_roll = euler1[0]
            true_pitch = euler1[1]
            true_yaw = euler1[2]
            
    if file.endswith(".png"):
        # load the image and get the euler angles from the QuaternionEstimator class
        img = cv2.imread(os.path.join(path, file) , cv2.IMREAD_GRAYSCALE)
        euler = QuaternionEstimator1().predict1(img)
        print(euler)
        # save the euler angles to the csv file in a way that the true roll pitch and yaw can be obtained from the json files in the same folder and written in the same row as the estimated roll pitch and yaw
        roll = euler[0,0]
        pitch = euler[0,1]
        yaw = euler[0,2]
        flag = 1
        
    # write the estimated roll pitch and yaw and the true roll pitch and yaw to the csv file in the same row
    if flag == 1:
        
        with open('C:/Users/Aiden/Desktop/dummy-sat/MDA-Aiden/estimated_angles6.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([roll, pitch, yaw, true_roll, true_pitch, true_yaw])
            flag = 0
    







