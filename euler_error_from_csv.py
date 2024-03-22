import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

# function to calculate relative error in degrees
def relative_error(estimate, true):
    error = abs(estimate - true)
    if error > 180:
        error = 360 - error
    return error
estimated_roll = []
estimated_pitch = []
estimated_yaw = []
true_roll = []
true_pitch = []
true_yaw = []
roll_error = []
pitch_error = []
yaw_error = []

# open csv file and read euler angles
with open('estimated_angles5.csv', newline='') as file:
    reader = csv.reader(file)
    next(reader) # skip header
    for row in reader:
        # save the estimated roll pitch and yaw in an array
        estimated_roll.append(float(row[0]))
        estimated_pitch.append(float(row[1]))
        estimated_yaw.append(float(row[2]))
        # save the true roll pitch and yaw in an array
        true_roll.append(float(row[3]))
        true_pitch.append(float(row[4]))
        true_yaw.append(float(row[5]))
        
        # calculate the relative error in degrees
        roll_error.append(relative_error(float(row[0]), float(row[3])))
        pitch_error.append(relative_error(float(row[1]), float(row[4])))
        yaw_error.append(relative_error(float(row[2]), float(row[5])))
        
        



# Plot the distribution of relative error for roll, pitch, and yaw
plt.figure(figsize=[15,5])
plt.subplot(131)
plt.hist(roll_error, bins=30)
plt.xlabel('Relative Error (deg)')
plt.ylabel('Frequency')
plt.title('Roll')
plt.subplot(132)
plt.hist(pitch_error, bins=30)
plt.xlabel('Relative Error (deg)')
plt.ylabel('Frequency')
plt.title('Pitch')
plt.subplot(133)
plt.hist(yaw_error, bins=30)
plt.xlabel('Relative Error (deg)')
plt.ylabel('Frequency')
plt.title('Yaw')
plt.show()

# Calculate median and standard deviation of relative errors
roll_error_median = np.median(roll_error)
pitch_error_median = np.median(pitch_error)
yaw_error_median = np.median(yaw_error)

roll_error_std = np.std(roll_error)
pitch_error_std = np.std(pitch_error)
yaw_error_std = np.std(yaw_error)

# Print median and standard deviation of relative errors
print('Median relative error for roll: ', roll_error_median)
print('Median relative error for pitch: ', pitch_error_median)
print('Median relative error for yaw: ', yaw_error_median)

print('Standard deviation of relative error for roll: ', roll_error_std)
print('Standard deviation of relative error for pitch: ', pitch_error_std)
print('Standard deviation of relative error for yaw: ', yaw_error_std)

