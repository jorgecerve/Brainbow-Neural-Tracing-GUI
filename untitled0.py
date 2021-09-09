# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:37:30 2021

@author: jorge
"""

import numpy as np
import cv2 as cv2
import tifffile as tiff
from skimage.morphology import skeletonize, skeletonize_3d
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt
from math import sqrt
import scipy



# Source file

source_filename = 'tracing.tif'

images = tiff.imread(source_filename) 

images = np.array(images)


images = np.swapaxes(images,0,2)

# tiff.imsave('f2ff.tif', images)

im_plot = Image.fromarray(images[1])
plt.imshow(im_plot)
plt.show() 
        
imagesfinal = []     
num_imgs = images.shape[0]
  
for i in range(num_imgs):  
   
    img = images[i]  
    
    
    # MEDIAN_BLUR_ITERATIONS = 1
    # MEDIAN_BLUR_FILTER_SIZE = 3
    # for iteration in range(MEDIAN_BLUR_ITERATIONS):
    #     img = cv2.medianBlur(img, MEDIAN_BLUR_FILTER_SIZE) 
    
    img = img/255
    img = skeletonize(img)    
    img = img.astype("uint8")
    img =img*255
    
    
    imagesfinal.append(img)  
    
imagesfinal = np.array(imagesfinal)
imagesfinal = np.swapaxes(imagesfinal,2,0)
    
tiff.imsave('88w.tif', imagesfinal)    
    
    