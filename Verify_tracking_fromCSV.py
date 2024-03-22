# this code will be given an x, y, and z distance, and a speed.
# the x, y, and z distance corresponds to the distance that an object travels from the camera.
# the speed is the speed that the object travels at.
# the object will first travel away from the camera, which is the positive z direction.
# the object will then travel in the positive x direction.
# the object will then travel in the positive y direction.

# this code will return the x, y, and z position of the object at a given time with the given distances and speed.

# the x, y, and z distances are in centimeters.
# the speed is in mm/s.

# the x, y, and z distances are the distances that the object travels from the camera relative to its starting position.
import matplotlib.pyplot as plt
import csv
import numpy as np 

z = 40.0 # distance that the object travels in the positive z direction in cm relative to the camera
x = 40.0 # distance that the object travels in the positive x direction in cm relative to the camera
y = 40.0 # distance that the object travels in the positive y direction in cm relative to the camera

speed = 50 # speed that the object travels at in mm/s


# the x, y, and z positions of the object at a given time
x_pos = []
y_pos = []
z_pos = []
time = []

# compute the total time that the trip will take from the given distances and speed
total_time = (z + x + y) / (speed / 10)


# read the file 'C:/Users/Aiden/Desktop/dummy-sat/MDA-Aiden/coordinates.csv'
# the 1st column is x, the 2nd column is y, and the 3rd column is z and the 4th column is time in seconds
# x, y, and z are in centimeters

with open('C:/Users/Aiden/Desktop/dummy-sat/MDA-Aiden/coordinates.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        x_pos.append(float(row[0])/10)
        y_pos.append(float(row[1])/10)
        z_pos.append(float(row[2])/10)
        time.append(float(row[3]))

# with the known path and speed of the object, compute the actual x, y, and z positions of the object at each timestamp from the csv file. we want to do this in cm so we can use these computed values as ground truth to test the accuracy of the csv file x, y, and z positions.
# the x, y, and z positions of the object at a given time
x_pos_actual = []
y_pos_actual = []
z_pos_actual = []

# each actual position should be computed assuming that the first prediction in the csv file is the correct prediction, so this should be the first position of the object. the first time should be subtracted from the time in the csv file to get the time that has passed since the object started moving.

# the first time in the csv file is the time that the object starts moving
first_time = time[0]

# the first x, y, and z positions in the csv file are the first x, y, and z positions of the object
first_x = x_pos[0]
first_y = y_pos[0]
first_z = z_pos[0]


# compute the actual x, y, and z positions of the object at each timestamp from the csv file. z direction for 100cm, then x direction for 100cm, then y direction for 100cm

for i in range(len(time)):
    # if the time is less than the time that the object travels in the positive z direction, then the object is still in the positive z direction
    if time[i] - first_time < z / (speed / 10):
        x_pos_actual.append(first_x)
        y_pos_actual.append(first_y)
        z_pos_actual.append(first_z + (speed / 10) * (time[i] - first_time))
    # if the time is less than the time that the object travels in the positive z direction and the positive x direction, then the object is still in the positive x direction
    elif time[i] - first_time < (z + y) / (speed / 10):
        x_pos_actual.append(first_x )
        y_pos_actual.append(first_y+ (speed / 10) * (time[i] - first_time - z / (speed / 10)))
        z_pos_actual.append(first_z + z)
    # if the time is less than the time that the object travels in the positive z direction and the positive x direction and the positive y direction, then the object is still in the positive y direction
    elif time[i] - first_time < (z + x + y) / (speed / 10):
        x_pos_actual.append(first_x + (speed / 10) * (time[i] - first_time - z / (speed / 10) - y / (speed / 10)))
        y_pos_actual.append(first_y + y)
        z_pos_actual.append(first_z + z)
    # if the time is greater than the time that the object travels in the positive z direction and the positive x direction and the positive y direction, then the object is no longer moving
    else:
        x_pos_actual.append(first_x + x)
        y_pos_actual.append(first_y + y)
        z_pos_actual.append(first_z + z)

    
    

# plot the actual x, y, and z positions of the object at each timestamp and the positions from the csv file on the same 3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_pos, y_pos, z_pos, c='r', marker='o')
ax.scatter(x_pos_actual, y_pos_actual, z_pos_actual, c='b', marker='o')
# set the scale of the x axis to be from 0 to 100
ax.set_xlim(0, 100)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

# plot the error for the whole trip as a function of time, and use the x,y, and z predicted to compute teh distance between predicted and actual in cm
# the error for the whole trip as a function of time
error = []

for i in range(len(time)):
    error.append(((x_pos[i] - x_pos_actual[i]) ** 2 + (y_pos[i] - y_pos_actual[i]) ** 2 + (z_pos[i] - z_pos_actual[i]) ** 2) ** (1/2))


avg_error = 0
for i in range(len(time)):
    if time[i] < 13:
        avg_error += error[i]
avg_error = avg_error / len(time)
 
print(avg_error)

plt.plot(time, error)
plt.xlabel('Time (s)')
plt.ylabel('Error (mm)')
# plot the average error on the plot
plt.axhline(y=avg_error, color='r', linestyle='-')
plt.show()





















