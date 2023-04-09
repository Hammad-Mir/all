import cv2
import os
import keras
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense , Conv2D, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D, Input, Conv2DTranspose
from keras.utils import np_utils
from keras.optimizers import Adadelta, RMSprop,SGD
from keras.layers.normalization import BatchNormalization
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

###########################################  Encoder  ####################################################


def Encoder(input_img):

	Econv1_1 = Conv2D(16, (3, 3), activation='relu', padding='same', name = "block1_conv1")(input_img)
	Econv1_1 = BatchNormalization()(Econv1_1)
	Econv1_2 = Conv2D(16, (3, 3), activation='relu', padding='same',  name = "block1_conv2")(Econv1_1)
	Econv1_2 = BatchNormalization()(Econv1_2)
	pool1 = MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same', name = "block1_pool1")(Econv1_2)
	
	Econv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "block2_conv1")(pool1)
	Econv2_1 = BatchNormalization()(Econv2_1)
	Econv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "block2_conv2")(Econv2_1)
	Econv2_2 = BatchNormalization()(Econv2_2)
	pool2= MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block2_pool1")(Econv2_2)

	Econv3_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "block3_conv1")(pool2)
	Econv3_1 = BatchNormalization()(Econv3_1)
	Econv3_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "block3_conv2")(Econv3_1)
	Econv3_2 = BatchNormalization()(Econv3_2)
	pool3 = MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block3_pool1")(Econv3_2)

	encoded = Model(inputs = input_img, outputs = pool3 )

	return encoded

##########################################  Decoder   ##################################################

def Decoder(inp ):

	up1 = Conv2DTranspose(128,(3,3),strides = (2,2), activation = 'relu', padding = 'same', name = "upsample_1")(inp)
	up1 = BatchNormalization()(up1)
	#up1 = merge([up1, inp[3]], mode='concat', concat_axis=3, name = "merge_1")
	Upconv1_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "Upconv1_1")(up1)
	Upconv1_1 = BatchNormalization()(Upconv1_1)
	Upconv1_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "Upconv1_2")(Upconv1_1)
	Upconv1_2 = BatchNormalization()(Upconv1_2)

	up2 = Conv2DTranspose(64,(3,3),strides = (2,2), activation = 'relu', padding = 'same', name = "upsample_2")(Upconv1_2)
	up2 = BatchNormalization()(up2)
	#up2 = merge([up2, inp[2]], mode='concat', concat_axis=3, name = "merge_2")
	Upconv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "Upconv2_1")(up2)
	Upconv2_1 = BatchNormalization()(Upconv2_1)
	Upconv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "Upconv2_2")(Upconv2_1)
	Upconv2_2 = BatchNormalization()(Upconv2_2)
	
	up3 = Conv2DTranspose(16,(3,3),strides = (2,2), activation = 'relu', padding = 'same', name = "upsample_3")(Upconv2_2)
	up3 = BatchNormalization()(up3)
	#up3 = merge([up3, inp[1]], mode='concat', concat_axis=3, name = "merge_3")
	Upconv3_1 = Conv2D(16, (3, 3), activation='relu', padding='same', name = "Upconv3_1")(up3)
	Upconv3_1 = BatchNormalization()(Upconv3_1)
	Upconv3_2 = Conv2D(16, (3, 3), activation='relu', padding='same', name = "Upconv3_2")(Upconv3_1)
	Upconv3_2 = BatchNormalization()(Upconv3_2)
	   
	decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name = "Ouput_layer")(Upconv3_2)
	convnet = Model(inputs = inp, outputs =  decoded)

	return convnet

##########################################################################################################

input_img = Input(shape=input_shape)

#Encoders[1]*100))
encoded = Encoder(input_img)	#return encoded representation with intermediate layer Pool3(encoded)

#Decoder
decoded = Decoder(Input(shape=(28, 28, 128)))

output_img = decoded(encoded(input_img))

model= Model(inputs = input_img, outputs = output_img )

model.summary()

model.compile(optimizer = SGD(0.0002), loss='binary_crossentropy', metrics = ["accuracy"])

model.fit(X_train, X_train, batch_size=50, epochs=150, verbose=1, validation_data=(X_val, X_val))

scores = model.evaluate(X_test, X_test, verbose=2)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

encoded.save_weights('enc.h5')