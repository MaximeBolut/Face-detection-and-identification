import h5py 
import os
import numpy as np
from numpy import genfromtxt
from fr_utils import *

from keras.models import Sequential,load_model
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')

import tensorflow as tf

from inception_blocks_v2 import *


import skimage
from skimage import io, data # to read and load images
from skimage.transform import resize
from skimage.feature import Cascade

import matplotlib.pyplot as plt  # to show images
import matplotlib.patches as patches


#Here is the image you want to use to detect, extract and identify face(s)... 
filename = os.path.join('C:\\coding\\image processing', 'mama.jpg')
#filename=input(print("give the image you want to analyse'))
image = io.imread(filename)


#load pre trained file from the module root
trained_file= data.lbp_frontal_face_cascade_filename()

#initialize the detector cascade
detector = Cascade(trained_file)

#apply detector on the image
detected = detector.detect_multi_scale(img=image, scale_factor=1.2, step_ratio=1, min_size=(50,50), max_size=(1000,1000))
#scale factor: search window mulitiply by this factor for the different search (from min size to max size)
#step ratio the step in for the search window to be moved
#min_size and max_size of the face to be detected (in px)
#detector return the detected face: [{'r':int, 'c':int , 'width': int, 'height':int}]
# r is top left corner row px, c is the top left corner colum px and width , height are self explenatory

def show_detected_face(image, detected):

    plt.imshow(image)
    img_desc=plt.gca()
    plt.set_cmap('gray')
    plt.title(label='face detected')
    plt.axis('off')

    for patch in detected:
        img_desc.add_patch(patches.Rectangle((patch['c'], patch['r']), patch['width'], patch['height'], fill=False, color = 'b', linewidth =2))

    plt.show()

def getFace(d, image):

  x,y=d['r'], d['c']
  width, height= (d['r']+d['width']), (d['c']+d['height'])
  #extract face
  face=image[x:width, y:height]
  
  return face


FRmodel = faceRecoModel(input_shape=(3, 96, 96))


# GRADED FUNCTION: triplet_loss
def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    Returns:
    loss -- real number, value of the loss
    """
   
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # Step 1: Compute the (encoding) distance between the anchor and the positive
    pos_dist =  tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
 
    return loss


#load pre trained model
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)


#database of registered people with their encoded face---PUT IMAGE OF PEOPLE YOU WANT TO IDENTIFY ( Front facing face, size of 96x96px) 
database = {}
database["maria"] = img_to_encoding("images/maria.jpg", FRmodel)
database["max"] = img_to_encoding("images/maxdata.jpg", FRmodel)


# GRADED FUNCTION: who_is_it
def who_is_it(image_path, database, model):
    """
    Implements face recognition by finding who is the person on the image_path image.
    
    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras
    
    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    
    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
    encoding = img_to_encoding(image_path, model)
    
    ## Step 2: Find the closest encoding ##
    
    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100
    
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current db_enc from the database. (≈ 1 line)
        dist = np.linalg.norm((encoding-db_enc))

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name.
        if dist < min_dist:
            min_dist = dist
            identity = name

    
    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        
    return min_dist, identity



show_detected_face(image, detected)
ax=[]
i=0
fig, ax=plt.subplots(ncols=len(detected), figsize=(8,6), sharex=True, sharey=True)
#for each detected face
for d in detected:
    #obtain the cropped face from detected coordinate
    face=getFace(d, image)
    face_scaled=resize(face,(96,96,3),anti_aliasing=True)
    io.imsave(str(i)+'.jpg', face_scaled)

    path="C:\\coding\\facedetect\\"+str(i)+'.jpg'
  
    min_dist, title = who_is_it(path, database, FRmodel)
    ax[i].imshow(face)
    ax[i].set_title(title + str(min_dist))
    ax[i].axis('off')
    i+=1



plt.show()



