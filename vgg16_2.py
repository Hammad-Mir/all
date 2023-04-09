import numpy as np
from PIL import Image
from numpy import * 
import os
import os.path
import cv2
import h5py
from keras.layers import Input, Dense
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.utils import np_utils
from keras.optimizers import SGD,RMSprop,adam
from keras import backend as K
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Reshape
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.convolutional import Convolution2D as Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import merge
from keras.optimizers import Adadelta, RMSprop
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16

seed = 7
np.random.seed(seed)

def data_loader(path_train):
   train_list0=os.listdir(path_train)
   
   # Map class names to integer labels
  # train_class_labels = { label: index for index, label in enumerate(class_names) } 
      
   # Number of classes in the dataset
   num_classes=len(train_list0)

    # Empty lists for loading training and testing data images as well as corresponding labels
   x=[]
   y=[]
   
   # Loading training data
   for label,elem in enumerate(train_list0):
           
           path1=path_train+'/'+str(elem)
           images=os.listdir(path1)
           for elem2 in images:
               path2=path1+'/'+str(elem2)
               # Read the image form the directory
               img = cv2.imread(path2)  
                #resize
               img2=cv2.resize(img, (224,224))
               # Append image to the train data list
               x.append(img2)
               # Append class-label corresponding to the image
               y.append(str(label))
    
        
   #Shuffling data

   c = list(zip(x,y))
   np.random.shuffle(c)
   x,y = zip(*c)

   # Convert lists into numpy arrays

   x = np.asarray(x)
   y = np.asarray(y)


   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

   return x_train,y_train,x_test,y_test

path_train='./Data/Medical_Challange_train/Comp'

X_train,y_train,X_test,y_test=data_loader(path_train)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
#print(y_train)

input_shape = (X_train.shape[1], X_train.shape[2],X_train.shape[3])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train / 255.
X_test = X_test / 255.

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


def VGG_16():

  input_img=Input(shape=input_shape)

  c1=Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)

  c1=Conv2D(64, (3, 3), activation='relu', padding='same')(c1)

  p1=MaxPooling2D((2, 2), strides=(2))(c1)

  c2=Conv2D(128, (3, 3), activation='relu', padding='same')(p1)

  c2=Conv2D(128, (3, 3), activation='relu', padding='same')(c2)

  p2=MaxPooling2D((2, 2), strides=(2))(c2)

  c3=Conv2D(256, (3, 3), activation='relu', padding='same')(p2)

  c3=Conv2D(256, (3, 3), activation='relu', padding='same')(c3)

  c3=Conv2D(256, (3, 3), activation='relu', padding='same')(c3)

  p3=MaxPooling2D((2, 2), strides=(2))(c3)

  c4=Conv2D(512, (3, 3), activation='relu', padding='same')(p3)

  c4=Conv2D(512, (3, 3), activation='relu', padding='same')(c4)

  c4=Conv2D(512, (3, 3), activation='relu', padding='same')(c4)

  p4=MaxPooling2D((2, 2), strides=(2))(c4)

  c5=Conv2D(512, (3, 3), activation='relu', padding='same')(p4)

  c5=Conv2D(512, (3, 3), activation='relu', padding='same')(c5)

  c5=Conv2D(512, (3, 3), activation='relu', padding='same')(c5)

  p5=MaxPooling2D((2, 2), strides=(2), name='last')(c5)

  f0=Flatten()(p5)

  f1=Dense(4096, activation='relu')(f0)

  f2=Dense(4096, activation='relu')(f1)

  pr=Dense(1000, activation='relu')(f2)

  v16=Model(input_img, p5)

  v16.load_weights('./Weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5')

  #v_16=Model(input_img, c5)

  return v16

def Pred():

  input_img=Input(shape=(7, 7, 512))

  f0=Flatten()(input_img)

  f1=Dense(4096, activation='relu')(f0)

  f2=Dense(1000, activation='relu')(f1)

  pr=Dense(2, activation='softmax')(input_img)

  pred=Model(input_img, pr)

  return pred

model1 = VGG_16()

model2 = Pred()

model = Model(input=model1.input, output=model2(model1.output))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



'''
model1.load_weights('./Weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5')

last_out = model1.get_layer('last').output

xx = keras.layers.Flatten()(last_out)

xx = keras.layers.Dense(1000, activation='relu')

xx = keras.layers.Dense(num_classes, activation='softmax')

model2 = Model(model1.input, xx)

model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
'''
#model1.summary()
'''
for layer in model.layer:
	trainable.layer = False
'''
# (5) Train
model.fit(X_train, y_train, batch_size=50, epochs=5, verbose=1, validation_data=(X_val, y_val))

scores = model.evaluate(X_test, y_test, verbose=2)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))