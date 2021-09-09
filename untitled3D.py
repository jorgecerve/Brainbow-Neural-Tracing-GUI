# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 18:53:30 2021

@author: jorge
"""
# FIRST: Convert tiff stacks to RGB manually with ImageJ!!!!!

import numpy as np
import cv2 as cv2
import tifffile as tiff
from skimage.morphology import skeletonize, skeletonize_3d
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt
from math import sqrt
import scipy

# Source file

source_filename = 'nTracer_rgb'
CONTRAST_FACTOR = 1.8
LOWER_THRESHOLD = 45
MEDIAN_BLUR_ITERATIONS = 20
MEDIAN_BLUR_FILTER_SIZE = 3

# source_filename = '2a_rgb'
# CONTRAST_FACTOR = 1.8
# LOWER_THRESHOLD = 15
# MEDIAN_BLUR_ITERATIONS = 10
# MEDIAN_BLUR_FILTER_SIZE = 3

# source_filename = '2b_rgb'
# CONTRAST_FACTOR = 1.8
# LOWER_THRESHOLD = 25
# MEDIAN_BLUR_ITERATIONS = 0
# MEDIAN_BLUR_FILTER_SIZE = 3

# source_filename = '3a_rgb'
# CONTRAST_FACTOR = 1.8
# LOWER_THRESHOLD = 12
# MEDIAN_BLUR_ITERATIONS = 0
# MEDIAN_BLUR_FILTER_SIZE = 3

# source_filename = '3c_rgb'
# CONTRAST_FACTOR = 1.8
# LOWER_THRESHOLD = 25
# MEDIAN_BLUR_ITERATIONS = 0
# MEDIAN_BLUR_FILTER_SIZE = 3

# source_filename = '5a_rgb'
# CONTRAST_FACTOR = 1.8
# LOWER_THRESHOLD = 25
# MEDIAN_BLUR_ITERATIONS = 0
# MEDIAN_BLUR_FILTER_SIZE = 3

# source_filename = '7b_rgb'
# CONTRAST_FACTOR = 3
# LOWER_THRESHOLD = 38
# MEDIAN_BLUR_ITERATIONS = 0
# MEDIAN_BLUR_FILTER_SIZE = 3

# source_filename = '8a_rgb'
# CONTRAST_FACTOR = 3
# LOWER_THRESHOLD = 25
# MEDIAN_BLUR_ITERATIONS = 0
# MEDIAN_BLUR_FILTER_SIZE = 3

# source_filename = '8c_rgb'
# CONTRAST_FACTOR = 3
# LOWER_THRESHOLD = 10
# MEDIAN_BLUR_ITERATIONS = 10
# MEDIAN_BLUR_FILTER_SIZE = 3


image = tiff.imread('NIH/RGB/' + source_filename + '.tif') 

UPPER_THRESHOLD = 255

DILATE_ITERATIONS = 1

rgb_weights = [0.2989, 0.5870, 0.1140]
luminance_weights = [0.299, 0.587, 0.114]

plots_flag = -1

def plot(img, title):  
    # Plot 50 iteration
    if (plots_flag == 50):
        im_plot = Image.fromarray(img)
        plt.imshow(im_plot)
        plt.title(title)
        plt.show() 

def contrast (img, factor):
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Contrast(img)    
    im_output = enhancer.enhance(factor)    
    img = np.array(im_output)   
    plot(img, "Contrast")
    return img
    
def greyscale(img):
    img = np.dot(img[...,:3], rgb_weights)
    img = img.astype("uint8") 
    plot(img, "Grayscale")
    return img

def luminance(img):
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]
    img = np. sqrt( (0.299*red)**2 + (0.587*green)**2 + (0.114*blue)**2 )   
    
    img = img.astype("uint8") 
    plot(img, "Luminance")
    return img

def threshold(img, low_threshold, up_threshold):
    ret, img = cv2.threshold(img, low_threshold, up_threshold, cv2.THRESH_BINARY) 
    plot(img, "Threshold")
    return ret, img

def medianFilter(img, filter_size, iterations):
    for iteration in range(iterations):
        img = cv2.medianBlur(img, filter_size) 
    plot(img, "Median Blur " + str(iterations) + " iterations")
    return img

def gaussianFilter(img, filter_size, iterations):
    for iteration in range(iterations):
        img = cv2.GaussianBlur(img, (3,3), 0)
    plot(img, "Gaussian Filter")
    return img

def squeletonize2D (img):
    img = img/255
    img = skeletonize(img)    
    img = img.astype("uint8")
    img = img*255 
    plot(img, "Squeletonize2D")
    return img

def dilate (img, iterations):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    img = cv2.dilate(img, kernel, iterations = iterations)  
    plot(img, "Dilate")
    return img

def erode (img, iterations):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    img = cv2.erode(img, kernel, iterations) 
    plot(img, "Erode")
    return img

def save_image(img):  
    img = Image.fromarray(img) 
    #img = img.resize((256, 256))       
    img.save("data/" + source_filename + "/images/image"+str(i)+".png")
    img = np.asarray(img)
    return img

def save_label(img):
    img = Image.fromarray(img)
    img.save("data/" + source_filename +"/labels/image"+str(i)+".png")
    
def save_combined_images(i):    
    images = [Image.open(x) for x in ["data/" + source_filename +"/images/image"+str(i)+".png", "data/" + source_filename +"/labels/image"+str(i)+".png"]]
    widths, heights = zip(*(i.size for i in images))    
    total_width = sum(widths)
    max_height = max(heights)    
    new_im = Image.new('RGB', (total_width, max_height))    
    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]    
    padding = str(i).rjust(4, '0')
    new_im.save('data/'+source_filename+'/'+'combined_'+source_filename+ '/'+ source_filename + '_'+ padding + '.png')

def mask_color_original(img, img_original):
    # create mask with same dimensions as image
    mask = np.zeros_like(img_original)    
    # copy your image_mask to all dimensions (i.e. colors) of your image
    for i in range(3): 
        mask[:,:,i] = img.copy()    
    # apply the mask to your image    
    img = cv2.bitwise_and(img_original, mask)
    plot(img, "Mask color")
    return img

num_imgs = image.shape[0]

images = []
imagesfinal = []

for i in range(num_imgs):   
    plots_flag = i
    
    img = image[i]           
    
    img = save_image(img)
    img_original = img
    
    # Resize
    # img_original = cv2.resize(img_original, dsize=(1000, 1000), interpolation=cv2.INTER_LINEAR)


    img = img.astype("uint8")
    plot(img, "Original")
    
    # Contrast image
    img = contrast (img, CONTRAST_FACTOR)  


    # Blur image
    #img = gaussianFilter(img, MEDIAN_BLUR_FILTER_SIZE, MEDIAN_BLUR_ITERATIONS)
    img = medianFilter(img, 
                       filter_size = MEDIAN_BLUR_FILTER_SIZE, 
                       iterations = MEDIAN_BLUR_ITERATIONS)
    
    
    # Convert to grayscale or luminance
    # img = greyscale(img)
    img = luminance(img) 
 
    # Binary threshold 
    ret, img = threshold(img, 
                     low_threshold = LOWER_THRESHOLD, 
                     up_threshold = UPPER_THRESHOLD)  
    
 
    # img = erode(img, 1)

    
    # Resize
    # img = cv2.resize(img, dsize=(1000, 1000), interpolation=cv2.INTER_LINEAR)     
    
    # Squeletonize 2D
    img = squeletonize2D (img)    
    
    # Dilate
    # img = dilate(img, 2)

    
    #mask with original
    # img = mask_color_original(img, img_original )
    
    
    # Save images
    images.append(img)  
    
    save_label(img)
    save_combined_images(i)
    
images = np.array(images)

#Swap axes
   
# Squeletonize 2D
#img = skeletonize_2d (img)

 





# Squeletonize 3D
#img = skeletonize_3d (img)

# Save tiff stack
tiff.imsave( source_filename +'_d2ff.tif', images)

############################################################


























