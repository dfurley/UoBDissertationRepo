#script for importing images and converting to correct format

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

#point to data
DATADIR = "/Volumes/Macintosh HD2/Temp_workspace/CNN_classifier/testing_data"
CATEGORIES = ["cyanobacteria","diatom","green"]

training_data = []
IMG_SIZE = 240
X = []
y = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category) #path to class dirs
        class_num = CATEGORIES.index(category) #assign category to numerical value, cyanobacteria = 0, diatom = 1, green = 2
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass

create_training_data()

random.shuffle(training_data)

for features, label in training_data:
    X.append(features)
    y.append(label)
    
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#save data for later use
import pickle

pickle_out = open("test_X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("test_y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()