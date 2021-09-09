# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 12:46:49 2021

@author: jorge
"""
#################################################################
import numpy as np
import cv2 as cv2
import tifffile as tiff
from skimage.morphology import skeletonize, skeletonize_3d
from PIL import Image, ImageEnhance, ImageTk
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog
from skimage.color import rgb2hsv, hsv2rgb
import colorsys
from os import remove

rgb_weights = [0.2989, 0.5870, 0.1140]
luminance_weights = [0.299, 0.587, 0.114]

MASTER1_IMG_SIZE = 220
MASTER2_IMG_SIZE = 500

filename = 0    

previous_no_clusters = 0
kmeans = 0


def contrast (img, factor):
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Contrast(img)    
    im_output = enhancer.enhance(factor)    
    img = np.array(im_output)  
    return img

def medianFilter(img, filter_size, iterations):
    for iteration in range(iterations):
        img = cv2.medianBlur(img, filter_size) 
    return img


def luminance(img):
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]
    img = np. sqrt( (0.299*red)**2 + (0.587*green)**2 + (0.114*blue)**2 )   
    
    img = img.astype("uint8") 
    return img

def threshold(img, low_threshold, up_threshold):
    ret, img = cv2.threshold(img, low_threshold, up_threshold, cv2.THRESH_BINARY) 
    return ret, img

def squeletonize2D (img):
    img = img/255
    img = skeletonize(img)    
    img = img.astype("uint8")
    img = img*255 
    return img

def squeletonize3D (img):
    img = img/255
    img = skeletonize_3d(img)    
    img = img.astype("uint8")
    return img

def dilate (img, iterations):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    img = cv2.dilate(img, kernel, iterations = iterations)  
    return img


def mask_color_original(img, img_original):
    # create mask with same dimensions as image
    mask = np.zeros_like(img_original)    
    # copy your image_mask to all dimensions (i.e. colors) of your image
    for i in range(3): 
        mask[:,:,i] = img.copy()    
    # apply the mask to your image    
    img = cv2.bitwise_and(img_original, mask)
    return img

def clustering (img, num_clusters):   

    # Get only HUE
    img = rgb2hsv(img)
    img = img[:,:,:,0]

    pixel_values = img.reshape((-1, 1))    
    pixel_values = np.float32(pixel_values) 
    pixel_values = pixel_values*180
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)   
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
    # masked_image = masked_image.reshape((-1, 3))    
    masked_image = masked_image.reshape((-1, 1)) 

    # color (i.e cluster) to disable
    if (num_cluster != -1 ):
        # masked_image[labels != num_cluster] = [0, 0, 0]   
        masked_image[labels != num_cluster] = 0
    return masked_image


def display_label(panel, img, row, column, height, width):
    imagenTK = Image.fromarray(img) 
    imagenTK = imagenTK.resize((height, width)) 
    imagenTK = ImageTk.PhotoImage(imagenTK)    
    label = tk.Label(panel, image=imagenTK)
    label.image = imagenTK    
    label.grid(row = row, column = column)      
    

# PIPELINE FOR VISUALIZATION
def pipeline(value):
    global previous_no_clusters
    
    # Get selected slice from stack and display it
    img_original = filename[no_slice.get()]    
    img = img_original    
    display_label(master1, img, 1, 0, MASTER1_IMG_SIZE, MASTER1_IMG_SIZE)    
    display_label(master2, img, 0, 0, MASTER2_IMG_SIZE, MASTER2_IMG_SIZE)
    
    if (apply_contrast.get()== 1):
        img = np.asarray(img)
        img = contrast(img, contrast_value.get())         
    display_label(master1, img, 1, 1, MASTER1_IMG_SIZE, MASTER1_IMG_SIZE)
        
    if (apply_blur.get()== 1):
        img = np.asarray(img)
        img = medianFilter(img, 
                           filter_size = int(np.ceil(blur_filter_size.get()) // 2 * 2 + 1), #Float to nearest odd
                           iterations = int(blur_iterations.get()))
    display_label(master1, img, 1, 2, MASTER1_IMG_SIZE, MASTER1_IMG_SIZE)
    
    if (apply_luminance.get()== 1):
        img = np.asarray(img)
        img = luminance(img) 
    display_label(master1, img, 1, 3, MASTER1_IMG_SIZE, MASTER1_IMG_SIZE)
        
    if (apply_threshold.get()== 1):
        img = np.asarray(img)
        ret, img = threshold(img, int(threshold_value.get()), 255)
    display_label(master1, img, 1, 4, MASTER1_IMG_SIZE, MASTER1_IMG_SIZE)
        
    if (apply_squeletonize2D.get()== 1):
        img = np.asarray(img)
        img = squeletonize2D (img) 
    display_label(master1, img, 1, 5, MASTER1_IMG_SIZE, MASTER1_IMG_SIZE)
        
    if (apply_dilate.get()== 1):
        img = np.asarray(img)
        img = dilate (img, int(dilate_value.get()) )
    display_label(master1, img, 1, 6, MASTER1_IMG_SIZE, MASTER1_IMG_SIZE)
        
    if (apply_colorMask.get()== 1):
        img = np.asarray(img)            
        img = mask_color_original(img, img_original)            
    display_label(master1, img, 1, 7, MASTER1_IMG_SIZE, MASTER1_IMG_SIZE)
    display_label(master2, img, 0, 1, MASTER2_IMG_SIZE, MASTER2_IMG_SIZE)
    
 
    # hist = histogram.histograma1D(filename)          
    # display_label(master2, hist, 0, 1, img.shape[0], img.shape[1])
    
    
    #PROBLEMA: EL CLUSTERING SE HACE CON TODO EL VOLUMEN 3D, PERO EL PIPELINE ES 1D
    # POR LO QUE DE MOMENTO HACE EL CLUSTERING CON EL ARCHIVO ORIGINAL, NO CON EL TRACING
    # if (apply_cluster.get()== 1):
    #     #img = np.asarray(img)
    #     if (previous_no_clusters != cluster_value.get()):            
    #         img, labels = clustering (filename, cluster_value.get()) 
    #         previous_no_clusters = cluster_value.get()
    #     display_label(img[no_slice.get()], 2, 7)  
        
    #     selection_cluster = 2
    #     img_cluster = select_cluster(img, selection_cluster, labels)
    #     img_cluster[labels == selection_cluster] = fluorescent[selection_cluster] 
    #     img_cluster = img_cluster.reshape(img.shape)   
    #     display_label(img_cluster[no_slice.get()], 2, 8)      
    print ('Pipeline END')  

def histogram1D (image):
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
    plt.bar(range(360), array, color=color)
    plt.savefig("tempp.png")
    img = cv2.imread("tempp.png")
    remove("tempp.png")
    return img        

# PIPELINE TO SAVE FINAL FILE
def process_all():
    imagesfinal = []
    
    img_original_stack = filename
    
    num_imgs = img_original_stack.shape[0] 
    for i in range(num_imgs):   
        
        img_original = img_original_stack[i]    
        img_original = Image.fromarray(img_original)    
        # img_original = img_original.resize((200, 200)) 
        img_original = np.asarray(img_original)
        img = img_original        
        
        if (apply_contrast.get()== 1):
            img = np.asarray(img)
            img = contrast(img, contrast_value.get())         
    
            
        if (apply_blur.get()== 1):
            img = np.asarray(img)
            img = medianFilter(img, 
                           filter_size = int(np.ceil(blur_filter_size.get()) // 2 * 2 + 1), #Float to nearest odd
                           iterations = int(blur_iterations.get()))
        
        if (apply_luminance.get()== 1):
            img = np.asarray(img)
            img = luminance(img) 
    
            
        if (apply_threshold.get()== 1):
            img = np.asarray(img)
            ret, img = threshold(img, int(threshold_value.get()), 255)
    
            
        if (apply_squeletonize2D.get()== 1):
            img = np.asarray(img)
            img = squeletonize2D (img) 
            
        if (apply_dilate.get()== 1):
            img = np.asarray(img)
            img = dilate (img, int(dilate_value.get()) )

            
        if (apply_colorMask.get()== 1):
            img = np.asarray(img)            
            img = mask_color_original(img, img_original)   


        # Save images
        imagesfinal.append(img)  
    
    # save_label(img)
    # save_combined_images(i)
    
    imagesfinal = np.array(imagesfinal)


    #imagesfinal = squeletonize3D(imagesfinal) 


    # Save tiff stack
    tiff.imsave('tracing.tif', imagesfinal)
    
    # Compute and show color histogram
    # histogram = histogram1D (imagesfinal)
    # display_label(master3, histogram, 0, 0, 432, 288)
    
    # KMEANS HERE!
    if (apply_cluster.get()== 1):                      
        number_clusters = cluster_value.get()          
        img, labels = clustering (imagesfinal, number_clusters)         
        tiff.imsave('clusterized.tif', img)        
        for selection_cluster in range(number_clusters):
            index = 180/number_clusters             
            img_cluster = select_cluster(img, selection_cluster, labels)
            img_cluster[labels == selection_cluster] = i * index+1
            img_cluster = img_cluster.reshape(img.shape)   
            tiff.imsave('clusters/clusterized_' + str(selection_cluster) +'.tif', img_cluster)

       

    

############## CHECKBOXES###########
def contrast_checkbox():
    if(apply_contrast.get()== 1):
        scale_contrast['state']= tk.NORMAL
    else:
        scale_contrast['state']= tk.DISABLED
    pipeline(-1)

def blur_checkbox():
    print(str(apply_blur.get()))
    pipeline(-1)
    
def luminance_checkbox():
    print(str(apply_luminance.get()))
    pipeline(-1)

def threshold_checkbox():
    print(str(apply_threshold.get()))
    pipeline(-1)

def squeletonize2D_checkbox():
    print(str(apply_squeletonize2D.get()))
    pipeline(-1)

def colorMask_checkbox():
    print(str(apply_colorMask.get()))       
    pipeline(-1) 

def cluster_checkbox():
    print(str(apply_cluster.get()))       
    pipeline(-1) 

def dilate_checkbox():
    print(str(apply_dilate.get()))       
    pipeline(-1) 
    


# Root panel
master = tk.Tk()
master.title('Neuron tracing editor. Jorge Cervera Perez')

master1 = tk.Frame(master)
master1.pack(side="top")

master2 = tk.Frame(master)
master2.pack(side="left")

master3 = tk.Frame(master)
master3.pack(side="right")

# v = tk.Scrollbar(master)
# v.pack(side = tk.RIGHT, fill = tk.Y)
        


# Variables
no_slice = tk.IntVar()

contrast_value = tk.DoubleVar()
apply_contrast = tk.IntVar(value=1)

blur_filter_size = tk.DoubleVar()
blur_iterations = tk.DoubleVar(value=5)
apply_blur = tk.IntVar(value=1)

apply_luminance = tk.IntVar(value=1)

threshold_value = tk.DoubleVar(value=45)
apply_threshold = tk.IntVar(value=1)

apply_squeletonize2D = tk.IntVar(value=1)

apply_colorMask = tk.IntVar(value=1)

dilate_value = tk.IntVar(value=0)
apply_dilate = tk.IntVar(value=0)

cluster_value = tk.IntVar(value=5)
apply_cluster = tk.IntVar(value=0)




# UPLOAD CALLBACK
def UploadAction(event=None):
    global filename
    filename = filedialog.askopenfilename()
    print('Selected:', filename)
    
    # Create a photoimage object of the image in the path
    img = tiff.imread(filename)  
    filename = img
    num_imgs = img.shape[0] 
        
    scale_no_slice = tk.Scale(master1, from_= 0, to=num_imgs-1, orient=tk.HORIZONTAL, 
                                  label='Image Stack',
                                  variable = no_slice, command = pipeline)    
    scale_no_slice.grid(row=2, column=0, sticky="nsew")

    pipeline(-1)
    
    # temporary for visualization
    # histogram = cv2.imread('out.png')
    # display_label(master3, histogram, 0, 0, 432, 288)
    

######################## BUTTONS#################################

#################### Import file button ##############################
button = tk.Button(master1, text='Import File', command=UploadAction)
button.grid(row = 0, column = 0)


#################### Contrast ##############################
f1 = tk.Frame(master1)
scale_contrast = tk.Scale(f1, from_= 1, to=6, orient=tk.HORIZONTAL, label='Contrast', 
                          variable = contrast_value, command = pipeline)
chek_contrast = tk.Checkbutton(f1, text='Apply Contrast', width=25,  
                               command=contrast_checkbox, variable= apply_contrast)
f1.grid(row=2, column=1, sticky="nsew")
scale_contrast.pack(side="top")
chek_contrast.pack(side="top")
                        
#################### Median Blur ##############################
f1 = tk.Frame(master1)
scale_blur_iterations = tk.Scale(f1 , from_= 0, to=20, orient=tk.HORIZONTAL, 
                                 label='Median Blur Iterations',
                                 variable = blur_iterations, command = pipeline)
scale_blur_filter_size = tk.Scale(f1 , from_= 2, to=15, orient=tk.HORIZONTAL, 
                                  label='Filter size',
                                  variable = blur_filter_size, command = pipeline)
chek_blur = tk.Checkbutton(f1, text='Apply Median Blur', width=25, 
                           command=blur_checkbox, variable= apply_blur)
f1.grid(row=2, column=2, sticky="nsew")
scale_blur_iterations.pack(side="top")
scale_blur_filter_size.pack(side="top")
chek_blur.pack(side="top")

############################## Luminance ##############################
f1 = tk.Frame(master1)
chek_luminance = tk.Checkbutton(f1, text='Apply Luminance', width=25, 
                                command=luminance_checkbox, 
                                variable= apply_luminance)
f1.grid(row=2, column=3, sticky="nsew")
chek_luminance.pack(side="top")

#################### Threshold ##############################
f1 = tk.Frame(master1)
scale_threshold = tk.Scale(f1, from_= 0, to=255, orient=tk.HORIZONTAL, label='Lower threshold', 
                          variable = threshold_value, command = pipeline)

chek_threshold = tk.Checkbutton(f1, text='Apply Threshold', width=25,  
                               command=threshold_checkbox, variable= apply_threshold)

f1.grid(row=2, column=4, sticky="nsew")
scale_threshold.pack(side="top")
chek_threshold.pack(side="top")

#################### squeletonize2D ##############################
f1 = tk.Frame(master1)
chek_squeletonize2D = tk.Checkbutton(f1, text='Apply squeletonize2D', width=25, 
                                command=squeletonize2D_checkbox, 
                                variable= apply_squeletonize2D)
f1.grid(row=2, column=5, sticky="nsew")
chek_squeletonize2D.pack(side="top")
#################### Dilate ##############################
f1 = tk.Frame(master1)
scale_dilate = tk.Scale(f1, from_= 1, to=10, orient=tk.HORIZONTAL, label='Dilate times', 
                          variable = dilate_value, command = pipeline)


chek_dilate = tk.Checkbutton(f1, text='Apply Dilate', width=25,  
                               command=dilate_checkbox, variable= apply_dilate)


f1.grid(row=2, column=6, sticky="nsew")
scale_dilate.pack(side="top")
chek_dilate.pack(side="top")

#################### ColorMask ##############################
f1 = tk.Frame(master1)
chek_colorMask = tk.Checkbutton(f1, text='Apply ColorMask', width=25, 
                                command=colorMask_checkbox, 
                                variable= apply_colorMask)


f1.grid(row=2, column=7, sticky="nsew")
chek_colorMask.pack(side="top")

#################### Clustering ##############################
f1 = tk.Frame(master3)
scale_cluster = tk.Scale(f1, from_= 1, to=20, orient=tk.HORIZONTAL, label='Centroids', 
                          variable = cluster_value, command = pipeline)


chek_cluster = tk.Checkbutton(f1, text='Apply Clustering (slow)', width=25,  
                               command=cluster_checkbox, variable= apply_cluster)

f1.grid(row=2, column=8, sticky="nsew")
scale_cluster.pack(side="top")
chek_cluster.pack(side="top")

########################## Apply all ####################################
button = tk.Button(master2, text='Save', width=25, command=process_all)
button.grid(row=1, column=1, sticky="nsew") 








master.mainloop()