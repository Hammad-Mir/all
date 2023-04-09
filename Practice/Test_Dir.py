import numpy as np
import cv2
import os
#from keras.models import Sequential
#from keras.layers import Dense , Conv2D, Dropout, Flatten
#from keras.utils import np_utils
#from sklearn.model_selection import train_test_split

def data_loader(path_train0, path_train1, path_train2):
   train_list0=os.listdir(path_train0)
   train_list1=os.listdir(path_train1)
   train_list2=os.listdir(path_train2)  

   print(train_list0)
   print(train_list1)
   print(train_list2)

path_train0='./Data/Medical_Challange_train/fold_0'
path_train1='./Data/Medical_Challange_train/fold_1'
path_train2='./Data/Medical_Challange_train/fold_2'

data_loader(path_train0, path_train1, path_train2)