'''Import the libraries'''
import os
import cv2
from keras.layers.core import *
from keras.layers import  Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose,ZeroPadding2D, Add
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential,load_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
import numpy as np
import scipy
import numpy.random as rng
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
#from skimage.transform import resize
#from skimage.io import imsave
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

'''Set Keras image format '''
K.set_image_data_format('channels_last')


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

	encoded = Model(input = input_img, output = [pool3, Econv1_2, Econv2_2, Econv3_2] )

	return encoded
#########################################  Bottleneck ##################################################
#
##
def neck(input_layer):

	Nconv = Conv2D(256, (3,3),padding = "same", name = "neck1" )(input_layer)
	Nconv = BatchNormalization()(Nconv)
	Nconv = Conv2D(128, (3,3),padding = "same", name = "neck2" )(Nconv)
	Nconv = BatchNormalization()(Nconv)

	neck_model = Model(input_layer, Nconv)
	return neck_model
#
##########################################  Decoder   ##################################################

def Decoder(inp ):

	up1 = Conv2DTranspose(128,(3,3),strides = (2,2), activation = 'relu', padding = 'same', name = "upsample_1")(inp[0])
	up1 = BatchNormalization()(up1)
	up1 = merge([up1, inp[3]], mode='concat', concat_axis=3, name = "merge_1")
	Upconv1_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "Upconv1_1")(up1)
	Upconv1_1 = BatchNormalization()(Upconv1_1)
	Upconv1_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "Upconv1_2")(Upconv1_1)
	Upconv1_2 = BatchNormalization()(Upconv1_2)

	up2 = Conv2DTranspose(64,(3,3),strides = (2,2), activation = 'relu', padding = 'same', name = "upsample_2")(Upconv1_2)
	up2 = BatchNormalization()(up2)
	up2 = merge([up2, inp[2]], mode='concat', concat_axis=3, name = "merge_2")
	Upconv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "Upconv2_1")(up2)
	Upconv2_1 = BatchNormalization()(Upconv2_1)
	Upconv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "Upconv2_2")(Upconv2_1)
	Upconv2_2 = BatchNormalization()(Upconv2_2)
	
	up3 = Conv2DTranspose(16,(3,3),strides = (2,2), activation = 'relu', padding = 'same', name = "upsample_3")(Upconv2_2)
	up3 = BatchNormalization()(up3)
	up3 = merge([up3, inp[1]], mode='concat', concat_axis=3, name = "merge_3")
	Upconv3_1 = Conv2D(16, (3, 3), activation='relu', padding='same', name = "Upconv3_1")(up3)
	Upconv3_1 = BatchNormalization()(Upconv3_1)
	Upconv3_2 = Conv2D(16, (3, 3), activation='relu', padding='same', name = "Upconv3_2")(Upconv3_1)
	Upconv3_2 = BatchNormalization()(Upconv3_2)
	   
	decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name = "Ouput_layer")(Upconv3_2)
	convnet = Model(input = inp, output =  decoded)
	return convnet

#########################################################################################################

##########################################'''Initialise the model.'''####################################

x_shape = 256
y_shape = 256
channels = 1
input_img = Input(shape = (x_shape, y_shape,channels))

#Encoder
encoded = Encoder(input_img)	#return encoded representation with intermediate layer Pool3(encoded), Econv1_3, Econv2_3,Econv3_3

#Decoder
HG_ = Input(shape = (x_shape/(2**3),y_shape/(2**3),128))
conv1_l = Input(shape = (x_shape,y_shape,16))
conv2_l = Input(shape = (x_shape/(2**1),y_shape/(2**1),64))
conv3_l = Input(shape = (x_shape/(2**2),y_shape/(2**2),128))
decoded = Decoder( [HG_, conv1_l, conv2_l, conv3_l])

#BottleNeck
Neck_input = Input(shape = (x_shape/(2**3), y_shape/(2**3),128))
neck = neck(Neck_input)

#Combined
output_img = decoded([neck(encoded(input_img)[0]), encoded(input_img)[1], encoded(input_img)[2], encoded(input_img)[3]])
model= Model(input = input_img, output = output_img )
model.summary()
model.compile(optimizer = Adam(0.0005), loss='binary_crossentropy', metrics = ["accuracy"])
#model.save_weights('Model_exp/UNet/Stats/UNet.h5')

#########################################################################################################


name = os.listdir("/media/biometric/Data21/Core_Point/data_used")
input_images = []
output_images = []

print("loading_images")
count = 0
for i in name : 
	if os.path.exists("/media/biometric/Data21/Core_Point/data_used/"+i) and os.path.exists("/media/biometric/Data21/Core_Point/Mask_gt/"+ i):
		img_x = cv2.imread("/media/biometric/Data21/Core_Point/data_used/"+i, 0)	
		img_x = cv2.resize(img_x, (256,256)) 
		print(i)
		img_x = img_x[:,:,np.newaxis]
		input_images.append(img_x)
		img_y = cv2.imread("/media/biometric/Data21/Core_Point/Mask_gt/"+ i, 0)
		img_y = cv2.resize(img_y, (256,256)) 
		img_y = img_y[:,:,np.newaxis]
		output_images.append(img_y)
'''
print("converting to numpy arrays")
#input_images = np.asarray(input_images, np.float32) /255 
#output_images = np.asarray(output_images, np.float32)/255
'''
print input_images[0].shape	
print("Data_splitting..")
X_train,X_test,Y_train,Y_test=train_test_split(input_images,output_images,test_size=0.001)
del input_images 
del output_images 

X_train = np.asarray(X_train, np.float16)/255
print("Done")
X_test = np.asarray(X_test, np.float16)/255
print("Done")
Y_train = np.asarray(Y_train, np.float16)/255
print("Done")
Y_test = np.asarray(Y_test, np.float16)/255
print("Done")
saveModel = "Model_exp/UNet/Stats/UNet.h5"
#numEpochs = 100
batch_size = 8
num_batches = int(len(X_train)/batch_size)
print "Number of batches: %d\n" % num_batches
saveDir = 'Model_exp/UNet/Stats/'
loss=[]
val_loss=[]
acc=[]
val_acc=[]
epoch=0;
best_loss=1000
r_c=0

while epoch <1001 :
    
    history=model.fit(X_train, Y_train, batch_size=batch_size, epochs=1, validation_data=(X_test,Y_test), shuffle=True, verbose=1) 
	#print float(history.history['loss'][0])

    #print 'INSIDE LOOP'
    model.save_weights(saveModel)

    epoch=epoch+1
    print "EPOCH NO. : "+str(epoch)+"\n"
    loss.append(float(history.history['loss'][0]))
    val_loss.append(float(history.history['val_loss'][0]))
    acc.append(float(history.history['acc'][0]))
    val_acc.append(float(history.history['val_acc'][0]))
    loss_arr=np.asarray(loss)
    e=range(epoch)
    plt.plot(e,loss_arr)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Training Loss')	
    plt.savefig('Model_exp/UNet/Stats/Plot'+str(epoch)+'.png')	
    plt.close()
    loss1=np.asarray(loss)	
    val_loss1=np.asarray(val_loss)
    acc1=np.asarray(acc)
    val_acc1=np.asarray(val_acc)

    np.savetxt('Model_exp/UNet/Stats/Loss.txt',loss1)
    np.savetxt('Model_exp/UNet/Stats/Val_Loss.txt',val_loss1)
    np.savetxt('Model_exp/UNet/Stats/Acc.txt',acc1)
    np.savetxt('Model_exp/UNet/Stats/Val_Acc.txt',val_acc1)

    s=rng.randint(len(X_test))
    x_test=X_test[s,:,:,:]
    x_test=x_test.reshape(1,256,256,1)
    mask_img = model.predict(x_test)
    x_test = x_test.reshape(256,256)
    mask_img = mask_img.reshape((256,256))
    temp = np.zeros([256,256*2])
    temp[:,:256] = x_test[:,:]+mask_img[:,:]
    temp[:,256:256*2] = x_test[:,:]
    temp = temp*255
    mask_img=mask_img*255
    cv2.imwrite('Model_exp/UNet/' + str(epoch+1) + ".bmp", temp)
    cv2.imwrite('Model_exp/UNet/' + str(epoch+1) + ".png", mask_img)

print("training Done.")

