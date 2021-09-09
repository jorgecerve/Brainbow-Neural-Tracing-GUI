# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 18:37:10 2021

@author: jorge
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import colorsys
from skimage.color import rgb2hsv, hsv2rgb
from  PIL import Image

# 1D HISTOGRAM
image = tiff.imread('we.tif') 


# def histogram1D (image):
#     hsvImage = rgb2hsv(image)
#     hue = hsvImage[:,:,:,0]
#     hue = hue.flatten()  
#     hue = hue*360
#     hue = hue.astype(int)
#     hue = hue[hue != 0]
    
#     hue_colors = np.arange(360)/360
#     color = []
#     for i in range(360):
#         color.append(colorsys.hsv_to_rgb(hue_colors[i], 1, 1))
    

#     array, bin_edges  = np.histogram(hue, 360)
    
#     # plt.bar(range(359), array[1:])
#     plt.bar(range(360), array, color=color)
#     plt.show()
#     plt.savefig("out.png")
#     img = cv2.imread("test.jpg")
#     return img

# image = histogram1D (image)



hsvImage = rgb2hsv(image)
hue = hsvImage[:,:,:,0]
hue = hue.flatten()  
hue = hue*360
hue = hue.astype(int)
hue = hue[hue != 0]

hue_colors = np.arange(360)/360
color = []
for i in range(360):
    color.append(colorsys.hsv_to_rgb(hue_colors[i], 1, 1))


array, bin_edges  = np.histogram(hue, 360)

# plt.bar(range(359), array[1:])
plt.bar(range(360), array, color=color)
plt.savefig("out.png")
img = cv2.imread("out.png")





# 2D HISTOGRAM
# image = tiff.imread('we.tif') 
# hsvImage = rgb2hsv(image)
# hue = hsvImage[:,:,:,0]
# hue = hue.flatten()  

# sat = hsvImage[:,:,:,1]
# sat = sat.flatten()  

# # Creating bins
# x_min = np.min(hue)
# x_max = np.max(hue)  
# y_min = np.min(sat)
# y_max = np.max(sat)  
# x_bins = np.linspace(x_min, x_max, 50)
# y_bins = np.linspace(y_min, y_max, 20)
  
# # fig, ax = plt.subplots(figsize =(10, 7))
# # Creating plot
# plt.hist2d(hue[hue != 0], sat[hue != 0], bins =[x_bins, y_bins])


