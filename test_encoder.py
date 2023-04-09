import numpy as np
import cv2
import os
import keras
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
from sklearn.metrics import precision_score, recall_score, confusion_matrix


seed = 7
np.random.seed(seed)

def data_loader(path_test):
   test_list0=os.listdir(path_test)
   
   # Map class names to integer labels
  # train_class_labels = { label: index for index, label in enumerate(class_names) } 
      
   # Number of classes in the dataset
   num_classes=len(test_list0)

    # Empty lists for loading training and testing data images as well as corresponding labels
   x=[]
   y=[]
   
   # Loading training data
   for label,elem in enumerate(test_list0):
           
           path1=path_test+'/'+str(elem)
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


path_test='./Data/Medical_Challange_train/test_data'

x, y=data_loader(path_test)


x = x.astype('float32')
y_test = np_utils.to_categorical(y)


x = x/ 255.


model = load_model('encoder.h5')

#print(model.summary())

predict = model.predict(x, batch_size=50)

y_pred = np.argmax(predict, axis=1)

#print(y)
#print(y_pred)

def precision(y_true, y_predict):

  count =0
  count1 =0

  for i in range(len(y_predict)):
    if y_predict[i] == 1:
      if y_predict[i] == y_true[i]:
        count += 1

      else:
        count1 += 1

  return count/(count + count1)



PS = precision(y, y_pred)

print(PS)

def recall(y_true, y_predict):

  count =0
  count1 =0

  for i in range(len(y_true)):

    if y_true[i]==1:
      count1 += 1

  for i in range(len(y_predict)):

    if y_predict[i] == 1:

      if y_predict[i] == y_true[i]:
        count += 1

  return count/(count1)

RS = recall(y, y_pred)

print(RS)

CM = confusion_matrix(y, y_pred)

print(CM)

F_score = 2 *((PS * RS)/(PS + RS))

print(F_score)

scores = model.evaluate(x, y_test, verbose=2)
print("Baseline Error: %.2f%%" % scores[1])