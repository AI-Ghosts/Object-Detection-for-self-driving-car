"""
This sctipt implementing sliding window technique
The function called with different fixed window sizes



"""
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Sequential
from keras.models import load_model
from keras.models import model_from_json

VISIULIZATION = False
PREDICT_THRESHOLD = 0.5




classes=['car','ped','sign']

json_file = open(r'../../Model_Weights/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
classifier.load_weights(r"../../Model_Weights/self_driving_cars_weights.h5")


def Sliding_Window(img,step,siz,xstart,ystart,depth):
    detection=[[],[],[]]
    for y in range(0,img.shape[0],step):
        for x in range(0,img.shape[1],step):
            if(x+siz>img.shape[1] or y+siz>img.shape[0]):
                continue

            im=img[y:y+siz,x:x+siz]		#get subset (window) from the image 
            im2 = cv2.resize(im,(224,224))	#resize it for the CNN
            im2 = np.expand_dims(img_to_array(im2), axis=0)

            if(VISIULIZATION):
                clone = img.copy()
                cv2.rectangle(clone, (x, y), (x + siz, y + siz), (0, 255, 0), 2)
                cv2.imshow("Window", clone)
                cv2.waitKey(1)

            
            res = classifier.predict(im2)[0]
            idx = np.argmax(res)

            if(res[idx]>=PREDICT_THRESHOLD and classes[idx]=='car'):
                detection[idx].append((x+xstart,y+ystart,x+siz+xstart,y+siz+ystart,depth))
                if(not depth):
                	temp = Sliding_Window(im,int(step/1.5),int(siz/1.5),x+xstart,y+ystart,5)
                	for i in range(len(temp)):
                		detection[i].extend(temp[i])
    return detection