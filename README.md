Save_Images.py

This Python script uses the RealSense library to capture color and depth frames from a RealSense camera, and saves each frame as an image file to a specific directory on the user's computer. The script captures and saves 8 frames per second for a total of 2 seconds (starting after an initial delay of 2 seconds), with a timestamp associated with each frame. The script starts streaming from the RealSense camera and enters a loop to capture a total of 160 frames. However, the script only begins saving frames after 60 iterations, effectively introducing a delay of 2 seconds (at 30 frames per second) before the capture begins. 

Within each iteration of the loop, the script:

- Waits for a coherent pair of frames (color and depth) from the RealSense camera.
- Gets the current timestamp and converts it to a string.
- Displays the live color stream in a window titled 'RealSense'.
- Saves the color and depth frames as separate .png files, with the file name including both the frame number and the timestamp.

Improved_tracking2.py

This Python script contains two primary classes for object detection and center detection using the YOLOv5 model, as well as a helper function for visualizing the results. The two classes, ObjectDetector and CenterDetector, are designed for detecting a target object (e.g., spacecraft) and identifying the center of a cone within the object's bounding box, respectively. This is used for testing the tracking ability.The paths to the yolov5 weights must be changed to match their directory on your pc, as well as the weights from the 

Class: ObjectDetector

The ObjectDetector class is responsible for loading a YOLOv5 model for detecting target objects. It contains two main methods:

- _predict: This private method takes an input image and returns predictions in tensor format.
- process_image: This method processes an input image, returning a sequence of detections. It calls the _predict method internally.

Class: CenterDetector

The CenterDetector class is used for detecting the center of a cone in a cropped image, which is bound by the bounding box of the first detector. It also loads a YOLOv5 model specifically trained for center detection and contains two main methods:

- _predict1: This private method takes an input image and returns predictions in tensor format, similar to the _predict method in the ObjectDetector class.
- process_image1: This method processes an input image and returns the center of the cone within the bounding box. It calls the _predict1 method internally.

Function: draw_detection1

This helper function takes an input image and the detection results from either the ObjectDetector or CenterDetector classes, and draws a bounding box, label, and center dot on the input image. It can be used to visualize the detection results and ensure the accuracy of the object and center detection processes.

Verify_tracking_fromCSV.py

This script tests the accuracy of position tracking of an object moving in three dimensions. It generates an ideal trajectory based on the given inputs (distances in the x, y, and z direction, and speed), compares this with the actual trajectory recorded in a CSV file, and computes the error. 

The script operates under the following assumptions:
- The object initially moves away from the camera (positive z direction)
- Next, it moves in the positive x direction
- Finally, it moves in the positive y direction

- `x`, `y`, `z` : These are distances (in cm) that the object travels in the respective positive directions relative to the camera.
- `speed` : Speed of the object in mm/s.

The order of the directions, the speed, and the distances can easily be changed to accommodate the path that the robot takes. The error is calculated as the Euclidean distance between the actual and predicted positions. The speed and distances the object travels in each direction are hardcoded and must be modified based on the specific path of the robotic arm.


get_intrinsics.py 

This Python script is used to retrieve and print the intrinsic parameters of an Intel RealSense D435i camera. The intrinsic parameters include the focal length (in the x and y directions), the principal point, and the distortion coefficients.These values are used in the Improved_tracking2.py file in order to calculate the x,y,z position of the object. 

The script prints the following parameters:
- Depth Scale: This value is used to convert the values in the depth image into actual distances.
- fx: The focal length in the x direction, in pixels.
- fy: The focal length in the y direction, in pixels.
- ppx: The x-coordinate of the principal point, in pixels.
- ppy: The y-coordinate of the principal point, in pixels.

These values should be replaced in the Improved_tracking2.py file in order to accurately track the object. The current ones should be relatively accurate, but it's always good to run this script beforehand and check.


Usage:

Use Save_Images.py. Make sure that you change the save path to one on your pc. This will save depth and RGB images with timestamps.
Set the rgb_dir and depth_dir in Improved_tracking2.py to be the same as the path that the images are saved to in Save_Images.py. Also, change the save path for the excel csv file in Improved_tracking2.py to a path on your pc. 
Change the path of the excel file that is read in the Verify_tracking_fromCSV.py file to match the excel file created with Improved_tracking2.py. 
Modify the Verify_tracking_fromCSV.py file to match your path and speed for the robotic arms path, and this will calculate the relative error at each timestamp, and display the information in a convenient manner. 

detect_and_crop2.py


To utilize the utility for obtaining and saving real training data for the Quaternion CNN, follow these steps:


1. Prepare a directory containing the images you want to detect for training.
2. Provide the input directory and the output directory to store the cropped images in the detect_and_crop2.py file.
3. Based on the detected objects' bounding boxes, the code will automatically crop the images to the bounding box to the detected object.
4. The cropped images will be saved in the specified output directory, and can be used for adding real world training data for the quaternion network.

Verify_orientation_From_images.py
This script is designed to test the estimations obtained from the orientation networks using images generated by NVIDIA's Deep Data Synthesizer. This script analyzes the estimations and saves them along with the corresponding image names to a CSV file.


To use the script:


1. Ensure that the following files are present in the same directory as the script:
    - "Sat5_model.json": The JSON file containing the architecture of the orientation network.
    - "Sat5_model.h5": The weight file containing the trained weights of the orientation network.

2. Set the appropriate values for the following variables in the script:
    - "path": The path to the folder containing the generated images.
    - "estimated_angles_file": The path to the CSV file where the estimated angles will be saved.

The script performs the following tasks:

1. Loads the orientation network model from the "Sat5_model.json" and "Sat5_model.h5" files.

2. Iterates through each image file in the specified folder.

3. For each image, it estimates the roll, pitch, and yaw angles using the loaded model.

4. Retrieves the true roll, pitch, and yaw angles from the corresponding JSON file (assumed to be present in the same folder).

5. Saves the estimated roll, pitch, and yaw angles along with the true roll, pitch, and yaw angles to the specified CSV file.

6. Prints the estimated angles for each image during execution.

euler_error_from_csv.py

This script is designed to read the CSV file produced by Verify_orientation_From_images.py containing estimated and true roll, pitch, and yaw angles and perform an analysis of the errors. It calculates and displays the distribution of relative errors and provides statistics such as the median and standard deviation of the errors. To use the script Set the path to the CSV file containing the estimated and true roll, pitch, and yaw angles.

The script displays the following outputs:

1. Three histograms showing the distribution of relative errors for roll, pitch, and yaw angles. Each histogram represents the frequency of relative errors within specific bins.

2. The median relative error for roll, pitch, and yaw angles.

3. The standard deviation of relative errors for roll, pitch, and yaw angles.



Quaternion_Train2.py

This code implements a neural network for estimating quaternion orientation of the satellite and its features. It uses a dataset of PNG images with corresponding JSON file labels from the Nvidia deep data synthesizer. The neural network is trained to estimate the quaternion orientation of the satellite.

 Define the data directory and the save directory for the quaternion network in order to run the code. 

Crop-images4.py

This script is used to preprocess and prepare the images generated by the NVIDIA Deep Data Synthesizer for training the quaternion Convolutional Neural Network. The NVIDIA Deep Data Synthesizer is a tool for generating synthetic data, and this script helps in preparing the generated images for further training.

1. Set the INPUT_DIR to the directory path where the NVIDIA Deep Data Synthesizer generated images are stored.
2. Set the OUTPUT_DIR to the directory path where you want to save the processed images.
3. Set the IMAGE_SIZE to the desired dimensions for the resized images.
4. Run the script and the processed images will be saved to the specified OUTPUT_DIR.

Improved_tracking_live.py

This code is similar to improved_tracking2.py, with the only difference being that it is live, and it also performs the euler estimations. This is the main script to use in order to perform both the x,y,z estimations and the euler estimations live. The improved_tracking2.py has documentation on all of the classes and functions.






