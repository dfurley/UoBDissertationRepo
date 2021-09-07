# import the necessary packages
import numpy as np
import imutils
import os
import cv2
import argparse

#%%
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True,
	help="path to the required directory")
args = vars(ap.parse_args())

#%%
#assign directory path
path = args["directory"]
#set count variable
count = 0

#create new directory in current directory to save new images
os.mkdir(path + '/rotated')
newpath = path + '/rotated'

#%%
#loop over files in directory, looking for jpgs
for f in os.listdir(path):
    if f.endswith('.jpg'):
   
        #load image from directory
        i = cv2.imread(os.path.join(path,f))
        fn, fext = os.path.splitext(f)
        #loop over rotational angles without cutting off part of image, saving each image to the directory  
        for angle in np.arange(0,360,15):
            rotated = imutils.rotate_bound(i,angle)
            #resize image as 200x200
            resize = cv2.resize(rotated,(240,240))
            fn = fn + '_' + str(count).zfill(3)
            cv2.imwrite(os.path.join(newpath,'{}.jpg'.format(fn)),resize)
            fn, fext = os.path.splitext(f)
            count += 1       
        #reset count for next image
        count = 0