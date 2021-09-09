# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 17:45:54 2021

@author: jorge
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff

from skimage.color import rgb2hsv, hsv2rgb


source_filename = 'nTracer_rgb'
# image = tiff.imread('NIH/RGB/' + source_filename + '.tif') 
image = tiff.imread('we.tif') 


filename = image
NUM_CLUSTERS = 6

blue = [0, 111, 255]
blue_t = [19, 244, 239]
green = [104, 255, 0]
yellow = [250, 255, 0]
orange = [255, 191, 0]
red = [255, 0, 92]
pink = [255, 0, 92]
fluorescent = blue, blue_t, green, yellow, orange, red



plt.imshow(image[20])
plt.show()
    

# image = rgb2hsv(image)
# image = image[20,:,:,0]
# plt.imshow(image, cmap='hsv')


    
def clustering (img, num_clusters):
    
    # Get only HUE
    # img = rgb2hsv(img)

    pixel_values = img.reshape((-1, 3))    
    pixel_values = np.float32(pixel_values) 
    # pixel_values = pixel_values/255
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.8)   
    _, labels, (centers) = cv2.kmeans(pixel_values, num_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS )

    centers = np.uint8(centers)    
    labels = labels.flatten()    
    segmented_image = centers[labels.flatten()]   
    
    segmented_image = segmented_image.reshape(img.shape)  
    
    return segmented_image, labels



def select_cluster (img, num_cluster, labels):
    # disable only the cluster number 2 (turn the pixel into black)
    masked_image = np.copy(img)
    
    # convert to the shape of a vector of pixel values
    masked_image = masked_image.reshape((-1, 3))
    
    # color (i.e cluster) to disable    
    masked_image[labels != num_cluster] = [0, 0, 0]
    
    # convert back to original shape
    masked_image = masked_image.reshape(image.shape)    

    return masked_image




img, labels = clustering (filename, NUM_CLUSTERS) 
# img = hsv2rgb(img)
plt.imshow(img[20])
plt.title('Clustered Image')
plt.show()

for i in range(NUM_CLUSTERS):
    selection_cluster = i
    
    img_cluster = select_cluster(img, selection_cluster, labels)
    
    # img_cluster[labels == selection_cluster] = fluorescent[i]  
    # masked_image = masked_image.reshape(image.shape)  
      


    plt.imshow(img_cluster[20])
    plt.title('Clustered Image')
    plt.show()
    
    tiff.imsave('clusterized_' + str(selection_cluster) +'.tif', img_cluster)
















