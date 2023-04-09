import numpy as np
import cv2
import os
import keras
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
from sklearn.metrics import precision_score, recall_score


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
               y.append(int(label))
 
   # Convert lists into numpy arrays

   x = np.asarray(x)
   y = np.asarray(y)

   return x,y


path_train='./Data/Medical_Challange_train/test_data'

x, y=data_loader(path_train)


x = x.astype('float32')
y_test = np_utils.to_categorical(y)


x = x/ 255.


model = load_model('VGG16net.h5')

#print(model.summary())

predict = model.predict(x, batch_size=50)

y_pred = np.argmax(predict, axis=1)

print(type(y[0]))
print(type(y_pred[0]))

PS = precision_score(y, y_pred)

print(PS)

RS = recall_score(y, y_pred)

print(RS)

scores = model.evaluate(x, y_test, verbose=2)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))