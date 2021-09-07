#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:34:00 2019

@author: David
"""

# import the necessary packages
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

#create new directory in current directory to save new images
os.mkdir(path + '/grayscale')
newpath = path + '/grayscale'

#%%
#loop over files in directory, looking for jpgs
for f in os.listdir(path):
    if f.endswith('.jpg'):
   
        #load image from directory
        image = cv2.imread(os.path.join(path,f))
        fn, fext = os.path.splitext(f)
        
        gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #convert to grayscale
        resize = cv2.resize(gray_image,(240,240))
        cv2.imwrite(os.path.join(newpath,'{}.jpg'.format(fn)),resize) #write to new folder