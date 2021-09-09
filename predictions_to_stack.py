# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 12:46:42 2021

@author: jorge
"""

import os 
import cv2
import numpy as np
import tifffile as tiff

path = 'predictions/'

images_names = os.listdir(path)
images = []

for img in images_names:

    img = path+img
    
    img = cv2.imread(img)
    img = img.astype("uint8")
    images.append(img) 
    

    
images = np.array(images)


# Save tiff stack
tiff.imsave( 'predictions.tif', images)