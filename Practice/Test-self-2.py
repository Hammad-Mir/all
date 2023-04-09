import numpy as np
import cv2
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

seed = 7
np.random.seed(seed)

def data_loader(path_train0):
   train_list0=os.listdir(path_train0)
   
  # Map class names to integer labels
  # train_class_labels = { label: index for index, label in enumerate(class_names) } 
      
   # Number of classes in the dataset
   num_classes=len(train_list0)

    # Empty lists for loading training and testing data images as well as corresponding labels
   x=[]
   y=[]
   
   # Loading training data
   for label,elem in enumerate(train_list0):
           
           path1=path_train0+'/'+str(elem)
           images=os.listdir(path1)
           for elem2 in images:
               path2=path1+'/'+str(elem2)
               # Read the image form the directory
               img = cv2.imread(path2)  
                #resize
               img2=cv2.resize(img, (256,256))
               # Append image to the train data list
               x.append(img2)
               # Append class-label corresponding to the image
               y.append(str(label))
                
   # Convert lists into numpy arrays
   x = np.asarray(x)
   y = np.asarray(y)
   
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

   return x_train,y_train,x_test,y_test

path_train0='./Data/Medical_Challange_train/fold_0'

#Calling Data Loader function
X_train,y_train,X_test,y_test=data_loader(path_train0)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
#print(y_train)

input_shape = (X_train.shape[1], X_train.shape[2],X_train.shape[3])

# forcing the precision of the pixel values to be 32 bit
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255.
X_test = X_test / 255.

# one hot encode outputs using np_utils.to_categorical inbuilt function
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

#Splitting the trining data into training and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

def baseline_model():

  model=Sequential()

  model.add(Conv2D(64, (5, 5), input_shape=input_shape, strides=(2,2), padding='valid', activation='relu'))

  model.add(Conv2D(64, (5, 5), strides=(2,2), padding='valid', activation='relu'))

  model.add(MaxPooling2D((2, 2), strides=(2, 2)))

  model.add(Conv2D(128, (5, 5), strides=(2,2), padding='valid', activation='relu'))

  model.add(Conv2D(128, (5, 5), strides=(2,2), padding='valid', activation='relu'))

  model.add(BatchNormalization())

  model.add(Conv2D(256, (3, 3), strides=(2,2), padding='valid', activation='relu'))

  model.add(Conv2D(256, (3, 3), strides=(2,2), padding='valid', activation='relu'))

  #model.add(MaxPooling2D((2, 2), strides=(1, 1)))

  #model.add(Conv2D(512, (3, 3), strides=(2,2), padding='valid', activation='relu'))

  #model.add(Conv2D(512, (3, 3), strides=(2,2), padding='valid', activation='relu'))

  #model.add(BatchNormalization())

  model.add(Flatten())

  model.add(Dense(4096, activation='relu'))

  model.add(Dropout(0.2))

  model.add(Dense(1000, activation='relu'))

  model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))

  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  print(model.summary())

  return model

model=baseline_model()

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=25, verbose=2)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))