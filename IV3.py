import numpy as np
import cv2
import os
from keras.models import Sequential, Model
from keras.layers import Dense , Conv2D, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
import keras
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.applications.inception_v3 import InceptionV3

#seed = 7
#np.random.seed(seed)

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


   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)

   return x_train,y_train,x_test,y_test

path_train='./Data/Medical_Challange_train/test_data'

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

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

p_model=InceptionV3(input_shape=input_shape, include_top=False, weights=None)

#p_model.load_weights('./Weights/inception_v3_weights_tf_dim_ordering_tf_kernels.h5')

last_layer = p_model.get_layer('mixed10')
print ('last layer output shape:', last_layer.output_shape)
last_output = last_layer.output

# Flatten the output layer to 1 dimension
xx = keras.layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
xx = keras.layers.Dense(1024)(xx)
xx = keras.layers.BatchNormalization()(xx)
xx = keras.layers.Activation('relu')(xx)
# Add a dropout rate of 0.2
#xx = keras.layers.Dropout(0.2)(xx)
# Add a final softmax layer for multi-class classification
xx = keras.layers.Dense(num_classes, activation='softmax')(xx)
transfer_model = Model(p_model.input, xx)

# Configure and compile the model
transfer_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

for layer in p_model.layers:
  layer.trainable = False

transfer_model.fit(X_train, y_train, batch_size=50, epochs=5, verbose=1, validation_data=(X_val, y_val))

scores = transfer_model.evaluate(X_test, y_test, verbose=2)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))