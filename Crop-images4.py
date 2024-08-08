#"""
#I have a folder in location 'C:\Users\Aiden\Desktop\dummy-sat\QuaternionTrainingImages\Cone\TestCapturer' that has 10000 PNG images labeled with names 000000 to 009999. each image has a corresponding JSON file with the same name labeled with the quaternion orientation of an object in the image, and the 2 pixel coordinates of the bounding box of the object in the image. Any images that include the word 'depth' in the name should be ignored. write a python code to do the following
#1. Given the folder 'C:\Users\Aiden\Desktop\dummy-sat\QuaternionTrainingImages\Cone\TestCapturer', take each json file and save the info in each file under "objects";  "class", and "quaternion_xyzw" to use in step 3. we also want the info under "bounding_box" "top_left" "bottom_right" info.
#2. each json file has a corresponding image with the same name. if the word depth is in the name, the image can be ignored. crop each image to the "bounding_box" info from above. check if the crop is valid, and if not, remove the image and JSON file data. save the image from valid crops to the folder 'C:\Users\Aiden\Desktop\dummy-sat\QuaternionTrainingImages\Cone-Cropped' with the same name plus the "class" name added ontto the name.
#3. make a new JSON file with the same name plus the "class" name added onto the end of both as the successfully cropped image, and write the "class", and "quaternion_xyzw" info from step 1 that correspond to the image into this JSON file. 
#4. this code should run in a loop that does this for each image one by one 
#"""

import os
import json
from PIL import Image

# get the path to the folder with the images and JSON files
path = 'F:\\pose\\TestCapturer_2023.03.14-14.21.35'

# get the path to the folder where we want to save the cropped images
save_path = 'F:\\Pose\\cropped3'

# get a list of all the files in the folder
files = os.listdir(path)

# loop through each file in the folder
for file in files:

    # if the file is a JSON file, open it and read the data
    if file.endswith('.json'):

        # open the JSON file and read the data
        with open(os.path.join(path, file)) as json_file:
            data = json.load(json_file)

        # get the name of the image from the JSON file name
        image_name = file[:-5] + '.png'

        # get the class of the object in the image from the JSON file data
        try:
            class_name = data['objects'][0]['class']

            # get the quaternion orientation of the object in the image from the JSON file data
            quaternion_xyzw = data['objects'][0]['quaternion_xyzw']

            # get the top left and bottom right coordinates of the bounding box of the object in the image from the JSON file data
            top_left = data['objects'][0]['bounding_box']['top_left']
            bottom_right = data['objects'][0]['bounding_box']['bottom_right']
            # if the image name does not contain 'depth', open it and crop it to the bounding box coordinates from above
            if 'depth' not in image_name:

                # open the image and crop it to the bounding box coordinates from above
                img = Image.open(os.path.join(path, image_name))
                img = img.crop((top_left[1], top_left[0], bottom_right[1], bottom_right[0]))

                # check if the crop is valid, and if not, remove the image and JSON file data
                if img.size == (0, 0):

                    # remove the image and JSON file data
                    os.remove(os.path.join(path, image_name))
                    os.remove(os.path.join(path, file))

                # if the crop is valid, save the image to the save folder with the class name added to the end of the name
                else:

                    # save the image to the save folder with the class name added to the end of the name
                    img.save(os.path.join(save_path, image_name[:-4] + '-' + class_name + '.png'))

                    # make a new JSON file with the same name plus the "class" name added onto the end of both as the successfully cropped image, and write the "class", and "quaternion_xyzw" info from step 1 that correspond to the image into this JSON file.
                    with open(os.path.join(save_path, file[:-5] + '-' + class_name + '.json'), 'w') as json_file:
                        json.dump({'class': class_name, 'quaternion_xyzw': quaternion_xyzw}, json_file)

        # if the class name is not in the JSON file data, remove the image and JSON file data
        except IndexError:

            # remove the image and JSON file data
            os.remove(os.path.join(path, image_name))
            os.remove(os.path.join(path, file))