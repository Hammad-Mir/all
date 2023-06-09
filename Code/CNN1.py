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

   # Map class names to integer labels
  # train_class_labels = { label: index for index, label in enumerate(class_names) } 
      
   # Number of classes in the dataset
   num_classes=len(train_list)

    # Empty lists for loading training and testing data images as well as corresponding labels
   x=[]
   y=[]
   
   # Loading training data
   for label,elem in enumerate(train_list):
           
           path1=path_train+'/'+str(elem)
           images=os.listdir(path1)
           for elem2 in images:
               path2=path1+'/'+str(elem2)
               # Read the image form the directory
               img = cv2.imread(path2)  
                #resize
               img2=cv2.resize(img, (150,150))
               # Append image to the train data list
               x.append(img2)
               # Append class-label corresponding to the image
               y.append(str(label))
               
                
   # Convert lists into numpy arrays
   x = np.asarray(x)
   y = np.asarray(y)
   
   #splitting the data into training and validation data
   x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.3, random_state = 42)

   return x_train,y_train,x_val,y_val

path_train='../Data/Medical_Challange_train/fold_0'

#Calling Data Loader function
X_train,y_train,X_test,y_test=data_loader(path_train)

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
#print(y_train)

input_shape = (X_train.shape[1], X_train.shape[2],X_train.shape[3])

# forcing the precision of the pixel values to be 32 bit
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255.
X_val = X_val / 255.

# one hot encode outputs using np_utils.to_categorical inbuilt function
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_test)
num_classes = y_val.shape[1]

##Splitting the trining data into training and validation
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#print(y_train)

# define baseline model
#The model is a simple neural network with one hidden layer with the same number of neurons as there are inputs (784)
def baseline_model():
	
	# create model
	model = Sequential()
	
	#We will add a Convolution layer with 32 filters of 3x3, keeping the padding as same
	model.add(Conv2D(32, (10,10), strides = (1,1), padding = 'same' , input_shape=input_shape, activation='relu'))
	
	#Flatten the feature map
	model.add(Flatten())
	
	#Adding FC Layer
	model.add(Dense(100, activation='relu'))
	
	#A softmax activation function is used on the output
	#to turn the outputs into probability-like values and 
	#allow one class of the 10 to be selected as the model's output #prediction.
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# build the model
model = baseline_model()

# Fit the model
#The model is fit over 10 epochs with updates every 200 images. The test data is used as the validation dataset
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3, batch_size=10, verbose=2)

# Final evaluation of the model
scores = model.evaluate(X_val, y_val, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))