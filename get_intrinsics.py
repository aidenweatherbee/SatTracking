# this python code will be used to get the intrinsics of the intel realsense d435i camera
# the intrinsics are the focal length, the principal point, and the distortion coefficients

import pyrealsense2 as rs
import numpy as np

# setup the camera at 640 x 480 
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
 
# get the intrinsics of the camera. that means fx, fy, ppx, ppy 
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# print fx
intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
print("fx: ", intr.fx)
# print fy
print("fy: ", intr.fy)
# print ppx
print("ppx: ", intr.ppx)
# print ppy
print("ppy: ", intr.ppy)

