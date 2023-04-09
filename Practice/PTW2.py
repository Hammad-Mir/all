from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import numpy as np
import cv2
import os
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

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
   
   return x,y

path_train0='./Data/Medical_Challange_train/fold_0'

X_train,y_train = data_loader(path_train0)

print(X_train.shape)
print(y_train.shape)

input_shape = (X_train.shape[1], X_train.shape[2],X_train.shape[3])

# forcing the precision of the pixel values to be 32 bit
X_train = X_train.astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255.

# one hot encode outputs using np_utils.to_categorical inbuilt function
y_train = np_utils.to_categorical(y_train)
num_classes = y_train.shape[1]

#Splitting the trining data into training and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
model_vgg16_conv.summary()

input = Input(shape=input_shape, name = 'image_input')

output_vgg16_conv = model_vgg16_conv(input)

x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(num_classes, activation='softmax', name='predictions')(x)

my_model = Model(input=input, output=x)

my_model.summary()

my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

my_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3, batch_size=5, verbose=2)