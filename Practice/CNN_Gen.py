from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
from keras.models import Sequential
from keras.layers import Dense , Conv2D, Dropout, Flatten
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

# fix random seed for reproducibility
#we always initialize the random number generator to a constant seed #value for reproducibility of results.
seed = 7
np.random.seed(seed)

# load data from the path specified by the user
def data_loader(path_train):
   train_list=os.listdir(path_train)  

   # Number of classes in the dataset
   num_classes=len(train_list)
   
   #array for data and labels
   x = []
   y = []
   
   #array with image names-complete_path and labels
   im_arr=[]
   
   # Loading training data
   for label,elem in enumerate(train_list):
           
           path1=path_train+'/'+str(elem)
           images=os.listdir(path1)
           for elem2 in images:
               path2=path1+'/'+str(elem2)
               
               im_arr.append([path2, label])
           
   return np.asarray(im_arr)

#path for the image directory
path_train='./Data/Medical_Challange_train/fold_0'

#retreiving image locations
array=data_loader(path_train)

np.random.shuffle(array)

#train, val = train_test_split(array, test_size = 0.2)
#print(array)

def generator(array, batch_size=200):
  
  num=len(array)
  
  for set in range(0, num, batch_size):
    
    batch_samples = array[set : set + batch_size]
    
    x_train=[]
    y_train=[]
    
    for sample in batch_samples:
      
      image = cv2.imread(str(sample[0]))
      
      y = sample[1]
      
      x_train.append(image)
      y_train.append(y)
      
        
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    
    x_train = x_train.astype('float32')
    x_train = x_train/255
    
    #y_train = np_utils.to_categorical(y_train,num_classes=2)
    #print(y_train)

    yield x_train,y_train

gen = generator(array)

def baseline_model():
  # create model
  model = Sequential()

  model.add(Conv2D(32, (5,5), strides=(1,1), padding='same', input_shape = (450, 450, 3), activation='relu'))
  
  model.add(Flatten())
  
  model.add(Dense(2, kernel_initializer='normal', activation = 'softmax'))
  
  # Compile model
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

# build the model
model = baseline_model()

loop = array.shape[0]//200
loop = int(loop)


for i in range(loop):
  x, y = next(gen)
  
  y = np_utils.to_categorical(y,num_classes=2)
  
  model.fit(x, y, epochs=1, batch_size = 200, verbose = 2, validation_split = 0.2)


#model.fit_generator(gen, steps_per_epoch = 12, epochs = 10, verbose = 2, )