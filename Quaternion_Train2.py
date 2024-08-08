

#1. import the necessary libraries.
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Nadam, Adam
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.image import smart_resize
#from keras.optimizers import Adam
# import relu from keras
from keras.layers import ReLU
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#2. use the PNG images with their JSON file labels in the folder "C:\Users\Aiden\Desktop\dummy-sat\QuaternionTrainingImages\All-Cropped" to save all of the images and "quaternion_xyzw" and "class" data that is contained in the JSON file label. each image contains a JSON file with the same name, and each image has a class that corresponds to one of 3 objects: "Cone1", "FullSolarPanel-colored" and "Docking_ring" that we would like to estimate the quaternion orientation of using the neural network we will create in step 3.
DATADIR = "F:\\pose\\cropped3" #the folder where all of the images are stored
#CATEGORIES = ["Cone1", "FullSolarPanel-colored", "Docking_ring"] #the classes of objects that we would like to estimate the quaternion orientation of using the neural network we will create in step 3.
IMG_SIZE = 128 #the size of the input image for the neural network is 64 by 64 pixels
training_data = [] #an empty list that will contain all of the training data for the neural network
#for category in CATEGORIES: #for each class of object that we would like to estimate the quaternion orientation of using the neural network we will create in step 3.
path = os.path.join(DATADIR) #the path to the folder where all of the images are stored for each class of object that we would like to estimate the quaternion orientation of using the neural network we will create in step 3.
#class_num = CATEGORIES.index(category) #the index number for each class of object that we would like to estimate the quaternion orientation of using the neural network we will create in step 3.
for img in os.listdir(path): #for each image in each folder for each class of object that we would like to estimate the quaternion orientation of using the neural network we will create in step 3.
    try: #try to do this:
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) #convert each image into an array with grayscale values from 0 to 255 for each pixel in each image for each class of object that we would like to estimate the quaternion orientation of using the neural network we will create in step 3.
        #img_array = cv2.imread(os.path.join(path,img))
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) #resize each image into an array with grayscale values from 0 to 255 for each pixel in each image for each class of object that we would like to estimate the quaternion orientation of using the neural network we will create in step 3. so that all images are 64 by 64 pixels in size.
        #new_array = smart_resize(img_array, (IMG_SIZE, IMG_SIZE), interpolation='bilinear')
        # convert the image to grayscale
        #gray = cv2.cvtColor(new_array, cv2.COLOR_BGR2GRAY)
        with open(os.path.join(path,img[:-4]+'.json')) as json_file: #open each JSON file with a name that is identical to its corresponding image file name except for its file extension which is ".json" instead of ".png". 
            data = json.load(json_file) #save all of the data from each JSON file with a name that is identical to its corresponding image file name except for its file extension which is ".json" instead of ".png". 
            quaternion_xyzw = data['quaternion_xyzw'] #save all of the "quaternion_xyzw" data from each JSON file with a name that is identical to its corresponding image file name except for its file extension which is ".json" instead of ".png". 
            training_data.append([new_array, quaternion_xyzw]) #append all of the training data from each JSON file with a name that is identical to its corresponding image file name except for its file extension which is ".json" instead of ".png". 
    except Exception as e: #if the try statement above fails to execute, then do this:
        pass #do nothing.

#3.Create the neural network that will be trained using the data prepared from step 2.the neural network should estimate "quaternion_xyzw" for each object "Class". the input image size is 64 by 64
X = [] #an empty list that will contain all of the training data for the neural network
y = [] #an empty list that will contain all of the training data for the neural network
for features, label in training_data:#for each image and its corresponding JSON file label in the folder "C:\Users\Aiden\Desktop\dummy-sat\QuaternionTrainingImages\All-Cropped"
    # use smart resize to resize the images to 128x128 and normalize the pixel values to be between 0 and 1
    X.append(features) #append all of the training data for the neural network
    y.append(label) #append all of the training data for the neural network
# use smart resize to resize the images to 128x128 and normalize the pixel values to be between 0 and 1
#for i in range(len(X)):
#    X[i] = smart_resize(X[i], (IMG_SIZE, IMG_SIZE), interpolation='bilinear')

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #convert all of the training data for the neural network into an array 
y = np.array(y) #convert all of the training data for the neural network into an array


X = X/255.0 #normalize all of the training data for the neural network

def quaternion_frobenius_norm_loss(y_true, y_pred):
    # y_true and y_pred are assumed to have the shape (batch_size, 4)
    # where the last dimension corresponds to the quaternion (x, y, z, w) 
    diff = K.square(y_true - y_pred)
    frobenius_norm = K.sqrt(K.sum(diff, axis=-1))
    return K.mean(frobenius_norm)


model = Sequential() #create a sequential model for the neural network
model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:])) #add a convolutional layer to the neural network with 32 filters, a 3 by 3 filter size and an input shape that is equal to the shape of X.
model.add(Activation('relu')) #add a relu activation function to the convolutional layer in the neural network.
model.add(MaxPooling2D(pool_size=(2, 2))) #add a max pooling layer to the neural network with a pool size of 2 by 2 pixels.
model.add(Conv2D(64, (3, 3))) #add a convolutional layer to the neural network with 32 filters and a 3 by 3 filter size.
model.add(Activation('relu')) #add a relu activation function to the convolutional layer in the neural network.
model.add(MaxPooling2D(pool_size=(2, 2))) #add a max pooling layer to the neural network with a pool size of 2 by 2 pixels.
model.add(Conv2D(128, (3, 3))) #add a convolutional layer to the neural network with 32 filters and a 3 by 3 filter size.
model.add(Activation('relu')) #add a relu activation function to the convolutional layer in the neural network.
model.add(MaxPooling2D(pool_size=(2, 2))) #add a max pooling layer to the neural network with a pool size of 2 by 2 pixels.
model.add(Conv2D(256, (3, 3))) #add a convolutional layer to the neural network with 32 filters and a 3 by 3 filter size.
model.add(Activation('relu')) #add a relu activation function to the convolutional layer in the neural network.
model.add(MaxPooling2D(pool_size=(2, 2))) #add a max pooling layer to the neural network with a pool size of 2 by 2 pixels.
model.add(Dropout(0.3)) #add a dropout layer to prevent overfitting in the convolutional layer in the neural network with 25% dropout rate.
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(512)) #add a dense layer to the neural network with 64 neurons in it.
model.add(Activation('relu')) #add a relu activation function to the dense layer in the neural network.
model.add(Dropout(0.3)) #add dropout to prevent overfitting in the dense layer in the neural network with 25% dropout rate.
model.add(Dense(256)) #add a dense layer to the neural network with 32 neurons in it.
model.add(Activation('relu')) #add a relu activation function to the dense layer in the neural network.
model.add(Dropout(0.2)) #add dropout to prevent overfitting in the dense layer in the neural network with 25% dropout rate.
model.add(Dense(64)) #add a dense layer to the neural network with 32 neurons in it.
model.add(Activation('relu')) #add a relu activation function to the dense layer in the neural network.
model.add(Dense(4)) #add a dense layer to the neural network with 4 neurons in it because we want to estimate 4 quaternion values for each object class in each image that we input into our neural network model.)
#model.add(Activation(quaternion_activation)) #add a quaternion activation function to the dense layer in the neural network.

#optimizer = Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0001)
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0002, amsgrad=True)
model.compile(loss=quaternion_frobenius_norm_loss, optimizer=optimizer, metrics=['mean_squared_error', 'accuracy']) #compile the neural network model using the quaternion loss function, the Adam optimizer and the mean squared error metric.
#model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0002), metrics=['mean_squared_error', 'accuracy'])


#model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.00025), 
model.summary() #print a summary of our model
#4. train the neural network. use the graphics card if possible.
checkpointer = ModelCheckpoint(filepath='F:/Pose-estimation/ConeImages/TrainedNetwork/Sat5_model.weights.best.hdf5', verbose=1, save_best_only=True) #save the best model weights to a file called "model.weights.best.hdf5"
history = model.fit(X, y, batch_size=30, epochs=30, validation_split=0.05, callbacks=[checkpointer], verbose=2, shuffle=True) #train our model using a batch size of 64, 100 epochs, 10% of our data as validation data and shuffle our data before each epoch

#Q: how do i choose what the number of neurons in each layer should be for a regression problem?
#A: the number of neurons in the output layer should be equal to the number of values that you want to estimate. in this case, we want to estimate 4 values for each object class in each image that we input into our neural network model. so the number of neurons in the output layer should be 4. the number of neurons in the input layer should be equal to the number of pixels in each image. in this case, each image is 64 by 64 pixels. so the number of neurons in the input layer should be 64*64 = 4096. the number of neurons in the hidden layers should be between the number of neurons in the input layer and the number of neurons in the output layer. so the number of neurons in the hidden layers should be between 4096 and 4. i chose 256 neurons for the first hidden layer and 128 neurons for the second hidden layer.

# #5. save the Neural network.
model_json = model.to_json() #save our model as a JSON file
with open("F:/Pose-estimation/ConeImages/TrainedNetwork/Sat5_model.json", "w") as json_file: #open a file called "model.json"
    json_file.write(model_json) #write our model as a JSON file
model.save_weights("F:/Pose-estimation/ConeImages/TrainedNetwork/Sat5_model.h5") #save our model weights as a h5 file


#7. plot the accuracy of the neural Network on two plots. A logarithemic scale and a regular scale.
plt.plot(history.history['accuracy']) #plot the accuracy of the neural Network on a regular scale.
plt.plot(history.history['val_accuracy']) #plot the accuracy of the neural Network on a regular scale for validation data (data that was not used for training).
plt.title('model accuracy') #set the title of this plot to "model accuracy".
plt.ylabel('accuracy') #set the y-axis label of this plot to "accuracy".
plt.xlabel('epoch') #set the x-axis label of this plot to "epoch".
plt.legend(['train', 'test'], loc='upper left') #add a legend to this plot with labels "train" and "test" in the upper left corner of this plot.
plt.show() #show this plot on our screen.
plt.yscale('log') #set the y-axis scale of this plot to logarithmic scale (log).
plt.plot(history.history['accuracy']) #plot the accuracy of the neural Network on a logarithmic scale (log).
plt.plot(history.history['val_accuracy']) #plot the accuracy of the neural Network on a logarithmic scale (log) for validation data (data that was not used for training).
plt.title('model accuracy') #set the title of this plot to "model accuracy".
plt.ylabel('accuracy') #set the y-axis label of this plot to "accuracy".
plt.xlabel('epoch') #set the x-axis label of this plot to "epoch".
plt.legend(['train', 'test'], loc='upper left') #add a legend to this plot with labels "train" and "test" in the upper left corner of this plot.
plt.show() #show this plot on our screen.


#8. plot the loss of the neural Network on two plots. A logarithemic scale and a regular scale.
plt.plot(history.history['loss']) #plot the loss of the neural Network on a regular scale (linear).
plt.plot(history.history['val_loss']) #plot the loss of the neural Network on a regular scale (linear) for validation data (data that was not used for training).
plt.title('quaternion model loss') #set the title of this plot to "model loss".
plt.ylabel('loss') #set the y-axis label of this plot to "loss".
plt.xlabel('epoch') #set the x-axis label of this plot to "epoch".
plt.legend(['train', 'test'], loc='upper left') #add a legend to this plot with labels "train" and "test" in the upper left corner of this plot.
plt.show() #show this plot on our screen.
plt.yscale('log') #set the y-axis scale of this plot to logarithmic scale (log).
plt.plot(history.history['loss']) #plot the loss of the neural Network on a logarithmic scale (log).
plt.plot(history.history['val_loss']) #plot the loss of the neural Network on a logarithmic scale (log) for validation data (data that was not used for training).
plt.title('quaterrnion model loss') #set the title of this plot to "model loss".
plt.ylabel('loss') #set the y-axis label of this plot to "loss".
plt.xlabel('epoch') #set the x-axis label of this plot to "epoch".
plt.legend(['train', 'test'], loc='upper left') #add a legend to this plot with labels "train" and "test" in the upper left corner of this plot..
plt.show() #show this plot on our screen

# plot the mean squared error of the neural Network on two plots. A logarithemic scale and a regular scale.
plt.plot(history.history['mean_squared_error']) #plot the mean squared error of the neural Network on a regular scale (linear).
plt.plot(history.history['val_mean_squared_error']) #plot the mean squared error of the neural Network on a regular scale (linear) for validation data (data that was not used for training).
plt.title('model mean squared error log') #set the title of this plot to "model mean squared error".
plt.ylabel('loss') #set the y-axis label of this plot to "loss".
plt.xlabel('epoch') #set the x-axis label of this plot to "epoch".
plt.legend(['train', 'test'], loc='upper left') #add a legend to this plot with labels "train" and "test" in the upper left corner of this plot.
plt.show() #show this plot on our screen.
plt.yscale('log') #set the y-axis scale of this plot to logarithmic scale (log).
plt.plot(history.history['mean_squared_error']) #plot the mean squared error of the neural Network on a regular scale (linear).
plt.plot(history.history['val_mean_squared_error']) #plot the mean squared error of the neural Network on a regular scale (linear) for validation data (data that was not used for training).
plt.title('model mean squared error') #set the title of this plot to "model mean squared error".
plt.ylabel('loss') #set the y-axis label of this plot to "loss".
plt.xlabel('epoch') #set the x-axis label of this plot to "epoch".
plt.legend(['train', 'test'], loc='upper left') #add a legend to this plot with labels "train" and "test" in the upper left corner of this plot..
plt.show() #show this plot on our screen



