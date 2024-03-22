import pyrealsense2 as rs
import numpy as np
import cv2

# i want to add a timer, and save 8 images per second, so 1 image every 0.125 seconds
import time


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)


# we want to save each frame as a png file
# so we need to create a folder to store them
# we can use the os module to do this
import os
# create a folder to store the images
folder = 'C:\\Users\\Aiden\\OneDrive\\Desktop\\MDA-Aiden\\saved_images\\video16'
if not os.path.exists(folder):
    os.makedirs(folder)




y = 0

# start streaming
pipeline.start(config)


try:
    # we want to capture 30 frames
    for i in range(160):
        if i> 60:


            # wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            # if start is defined, t
            # add a timestamp that represents the time since the first image was saved
            stamp = time.time()
            
            # change the timestamp to a string
            stamp = str(stamp)
            


            # display the live stream
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            cv2.imshow('RealSense', color_image)
            cv2.waitKey(1)


            # save the image
            cv2.imwrite('F:\\pose\\both\\rgb\\img' + str(i) + 'time: ' + stamp + '.png', color_image)
            # save the depth image
            cv2.imwrite('F:\\pose\\both\\depth\\img' + str(i) + 'time: ' + stamp + '.png', depth_image)
            

finally:
    pipeline.stop()
