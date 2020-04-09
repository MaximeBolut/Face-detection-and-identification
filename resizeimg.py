import cv2
import os
import glob


from skimage import io
from skimage.transform import resize
import numpy as np

 

def resize(file, count):
    
    img = cv2.imread(file)                              # Read image
    img=cv2.resize(img, (96,96))
    #cv2.imshow('max resize',img) 
    #cv2.waitKey(0)
    cv2.imwrite(str(count)+'.jpg',img)                  # Save image


#actually worst quality
def resize_better(file, count):
    img=io.imread(file)
    face_scaled=resize(img,(96,96,3),anti_aliasing=True)
    #face_scaled = np.rollaxis(face_scaled, 2, 0) #to put channel first
    io.imsave(str(count)+'resize.jpg',face_scaled)




images = glob.glob("C:\\coding\\facedetect\\image_to_resize\\*.jpg")
cpt=0

for image_name in images:
	cpt=cpt+1
	resize(image_name, cpt)