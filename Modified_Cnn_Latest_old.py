import numpy
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
import os
import os.path
import theano
from PIL import Image
from numpy import *
from keras.callbacks import ModelCheckpoint


from keras.layers.core import Reshape
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.convolutional import Convolution2D as Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import merge
from keras.optimizers import Adadelta, RMSprop
import os
import os.path
import numpy as np
from PIL import Image
from numpy import * 
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split



K.set_image_dim_ordering('tf')

# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

# input image dimensions
img_rows, img_cols = 64, 64


# number of channels
img_channels = 1

# number of epochs
epochs=10
#batch Size
batch_size1=32

target_names = ['class 1', 'class 2']

# Path to the folder that is containing all the folders correspond to different class
path = '/home/biometric/Dr_Aditya_Nigam/Cnn_Ear/Image_Class'


list_files = os.listdir(path) # List_Files contains names of different folders inside 
print (list_files)
number_class = len(list_files) # Number of classes
print (number_class) 
num_samples = 0
#Processing images for each class
global combine_matrix # declaration of variable
n_da_per_class = [] # create list for containing size(total number of images) of each class
check_first = 1

for class_n in list_files:
    class_list = os.listdir(path + '/'+ class_n)
    class_im = array(Image.open(path + '/'+ class_n + '/' + class_list[0])) # open one image to get size
    image_m,image_n = class_im.shape[0:2] # get the size of the one image in Class
    class_imnbr = len(class_list) # get the number of images corresponding to each class
    n_da_per_class.append(class_imnbr) # append size of folder in n_da_per_class list
    num_samples = num_samples + class_imnbr
    class_immatrix_n = array([array(Image.open(path + '/'+ class_n + '/' + class_im2)).flatten()
                            for class_im2 in class_list],'f')
    
    print (size(class_immatrix_n))
    if check_first == 1:
        check_first = 0
        combine_matrix = class_immatrix_n
    else:
        combine_matrix  = numpy.concatenate((combine_matrix, class_immatrix_n), axis=0)  # Combining each matrix
    

label=numpy.ones((num_samples,),dtype = int) # number of elements in label = number of samples
class_ind = 0; # initializing class label
number = 0
sum_number = 0
for data_p_c in n_da_per_class:  # for each image, label is assigned
    number = number + int(data_p_c)
    for i in range(sum_number, number):
        label[i] = class_ind
    class_ind = class_ind + 1
    sum_number = sum_number + int(data_p_c)

data,Label = shuffle(combine_matrix,label, random_state=2)
train_data = [data,Label]
print ((train_data)) 
(X, y) = (train_data[0],train_data[1])
# number of output classes
nb_classes = number_class

# STEP 1: split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0],img_rows, img_cols,1)



# Data preprocessing
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
mean1 = numpy.mean(X_train) # for finding the mean for centering  to zero
X_train -= mean1
X_test -= mean1
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


def model1():

    input_img = Input(shape=(64,64,1))
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
      
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
       
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
   
    
    encoder = Model(input_img, conv5)
    #autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    #encoder.compile(loss='mean_squared_error', optimizer=rms)
    encoder.load_weights('cnn1.hdf5')
    return encoder




def model2():
	input_img = Input(shape=(8,8,64))

	#conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(input_img)
	#conv4 = BatchNormalization()(conv4)
	#conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
	#conv4 = BatchNormalization()(conv4)

	#conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
	#conv5 = BatchNormalization()(conv5)
	#conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
	#conv5 = BatchNormalization()(conv5)

	x = Flatten()(input_img)
	x = Dense(1024, activation='relu')(x)
	x = Dense(128, activation='relu')(x)
	x = Dense(32, activation='relu')(x)
	predictions = Dense(nb_classes, activation='softmax')(x)
 
	decoder = Model(input_img, predictions)
	decoder.load_weights('cnn2.hdf5')
	#decoder.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return decoder




encoder = model1()
# this model maps an input to its reconstruction
decoder = model2()


model = Model(input = encoder.input,output = decoder(encoder.output))
ada = Adadelta(lr = 5.0, rho = 0.95, epsilon = 1e-08, decay = 0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.load_weights('weights_ear_Cnn.hdf5')
for layers in encoder.layers:
	layers.trainable=True
model.summary()
#model.load_weights("model_Ear_cnn.hd5")
learning_rate=0.01
decay_rate=learning_rate/epochs
momentum=0.6
sgd=SGD(lr=learning_rate,momentum=momentum,decay=decay_rate,nesterov=True)

y_train = y_train.reshape((-1, 1)) # -1 refers to unknown; here for each image , one coloumn vector is generated


# Model will be saved after each epoch if there is improvement in validation accuracy.
filepath="weights_ear_Cnn1.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

#Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),batch_size=batch_size1, nb_epoch=epochs, callbacks=callbacks_list, verbose=2)


encoder.save_weights('cnn1.hdf5')
decoder.save_weights('cnn2.hdf5')
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()








# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print ("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Confusion Matrix

from sklearn.metrics import classification_report,confusion_matrix

Y_pred = model.predict(X_test)
#print(Y_pred)
y_pred = numpy.argmax(Y_pred, axis=1)
#print(y_pred)
#p=model.predict_proba(X_test) # to predict probability
print(classification_report(numpy.argmax(Y_test,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(numpy.argmax(Y_test,axis=1), y_pred))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5 and saving the weights
model.save_weights("weights_ear_Cnn1.hdf5")
print("Saved model to disk")
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("weights_ear_Cnn1.hdf5")
print("Loaded model from disk")
# evaluate loaded model on test data
loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test, verbose=2)
print ("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))




