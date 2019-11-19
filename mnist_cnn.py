
# Description: This program uses Convolutional Neural Networks (CNN) 
#              to classify handwritten digits as numbers 0 - 9

#import the libraries
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

#Load the data and split it into train and test sets
(X_train,y_train), (X_test, y_test) = mnist.load_data()

#Get the image shape
print(X_train.shape)#60000 rows of 28x28 pixel images with a depth=1 which signifies the images are gray scale (8 bit integers giving 256 possible shades of gray from black to white)
print(X_test.shape)#10000 rows of 28x28 pixel images with a depth=1 which signifies the images are gray scale (8 bit integers giving 256 possible shades of gray from black to white)

#Take a look at the first image (at index=0) in the training data set as a numpy array, This shows the image as a series of pixel values
X_train[0]

#Print the image label
y_train[0]

#Show the image as a picture
plt.imshow(X_train[0])

#Reshape the data to fit the model
X_train = X_train.reshape(60000, 28,28,1)
X_test = X_test.reshape(10000, 28, 28, 1)

#One-Hot Encode target column (Y-Data sets) : Convert the labels into a set of 10 numbers to input into the neural network
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

#Print the new label
print(y_train_one_hot[0])

#Build the CNN model
model = Sequential()
# add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))#Convolution layer to extract features from the input image, and create 64 channels of ReLu 3x3 convolved features (NOTE: Rectifier Linear Unit aka (ReLu))
model.add(Conv2D(32, kernel_size=3, activation='relu'))#Convolution layer to extract features from the input image, and create 32 channels of ReLu 3x3 convolved features
model.add(Flatten())#Flattening  layer makes the image a linear array or 1D Array or 1D Vector to feed into and connect with the neural network
model.add(Dense(10, activation='softmax'))# The neural network last layer with 10 neurons and activation function softmax, 10 neurons used because we have 10 labels

#Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Train the model
#Batch: Total number of training examples present in a single batch (None = default 32)
#Epoch:The number of iterations when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.
#Fit: Another word for train
hist = model.fit(X_train, y_train_one_hot, validation_data=(X_test, y_test_one_hot), epochs=3)

#Visualize the models accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

#Show predictions as probabilites for the first 4 images in the test set
predictions = model.predict(X_test[:4])
predictions

#Print our predicitons as number labels for the first 4 images
print( np.argmax(predictions, axis=1))
#Print the actual labels
print(y_test[:4])

#Show the first 4 images as pictures
for i in range(0,4):
  image = X_test[i]
  image = np.array(image, dtype='float')
  pixels = image.reshape((28,28))
  plt.imshow(pixels, cmap='gray')
  plt.show()
