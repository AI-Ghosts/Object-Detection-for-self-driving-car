from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers import Dense

from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2



from Sliding_Windows import *
from Bounded_box import *
from lane_detection import *




if __name__ == "__main__":
	
	testingImage = r'test1.jpg'
	img=cv2.imread(r"../Testing_Images/"+testingImage)
	
	
	
	step,siz=50,150
	Y=img.shape[0]
	ytop,ybottom=int(Y/2.4),int(Y/1.25)
	detection = Sliding_Window(img[ytop:ybottom,:],step,siz,0,ytop,0)

	testingImage = r'test1.jpg'
	img=cv2.imread(r"../Testing_Images/"+testingImage)
	frame=img
	img1 = detect_lane(frame, alpha = 0.8, opacity = 0.3)

	
	boxes = get_centroid_rectangles(detection[0],img.shape)
	for (x,y,w,h) in boxes:
		cv2.rectangle(img1, (x, y), (w, h), (255, 0, 0), 3)
	cv2.imwrite(r'../Results/predict_car.jpg', img1)