# -*- coding: utf-8 -*-
"""
Created on Tue May  1 18:26:34 2018

@author: Abdulrhman Tarek

"""
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import argparse

last_img = 0
started = 0
last_left, last_right = 0, 0
RED, GREEN = [0,5,250], [0,250,10]
BLUE, LIGHTBLUE = [250,5,5], [255,150,0]

def lines_sort(lines):
    n = len(lines)
    is_sorted = False
    while not is_sorted:
        is_sorted = True
        for i in range(n-1):
            if lines[i][0][3] < lines[i + 1][0][3]:
                temp = lines[i].copy()
                lines[i] = lines[i+1]
                lines[i+1] = temp
                is_sorted = False
    
    return lines

def Slope(points):
    x1, y1, x2, y2 = points[0],points[1],points[2],points[3]
    return (y2-y1)/(x2-x1)


def Region_Of_Interest(img, a=0.917, b=0.57, c=0.125, d=0.43, h=0.62):
    ## Finding a suitable polygon that fits the lane
    imshape = img.shape
    im_x, im_y = imshape[1], imshape[0]
    
    bottom_right = np.array([im_x * a, im_y], dtype='int')
    top_right = np.array([im_x * b, im_y * h], dtype='int')
    bottom_left = np.array([im_x * c , im_y], dtype='int')
    top_left = np.array([im_x * d , im_y * h], dtype='int')
    poly = [np.array([bottom_left,top_left,top_right,bottom_right])]
    
    # Create an array represents this polygon and fill the polygon with 255
    roi_mask = np.zeros_like(img)
    cv2.fillPoly(roi_mask, poly, 255)
    
    # Region of Interest image is AND the edge-detected image with the ROI polygon
    return cv2.bitwise_and(img, roi_mask)


def HoughTransform(img):
    rho = 4
    theta = np.pi/180
    #threshold is minimum number of intersections in a grid for candidate line to go to output
    threshold = 30
    min_line_len = 100  # Minimum line to be recognized
    max_line_gap = 180  # Maximum gap to be connected
    
    #lines_matrix = np.zeros_like()
    lines = cv2.HoughLinesP(img, rho, theta, threshold, minLineLength=min_line_len, maxLineGap=max_line_gap)
    #for line in lines:
    return lines


def draw_lines(img, lines, color = [255,255,255], alpha = 0.2, height = 0.62, thick=3):
    global started, last_left, last_right
    max_right, max_left = [],[]
    L, R = 0, 0
    
    for line in lines:
        slope = Slope(line[0])
        # Slope is inverted because of image axis
        if slope > 0.4 and L<2:
            max_left.append(line[0])
            L+=1
        elif slope < -0.4 and R<2:
            max_right.append(line[0])
            R+=1
    
    # Avoiding errors
    if len(max_right)<1 or len(max_left)<1:
        print("No lane")
        return
    if len(max_right)==1:
        max_right.append(max_right[0])
    if len(max_left)==1:
        max_left.append(max_left[0])
    
    # Line of average two nearest lines
    left_line = [
        (max_left[0][0]+max_left[1][0])//2 ,
        (max_left[0][1]+max_left[1][1])//2 ,
        (max_left[0][2]+max_left[1][2])//2 ,
        (max_left[0][3]+max_left[1][3])//2 
    ]
    
    right_line = [
        (max_right[0][0]+max_right[1][0])//2 ,
        (max_right[0][1]+max_right[1][1])//2 ,
        (max_right[0][2]+max_right[1][2])//2 ,
        (max_right[0][3]+max_right[1][3])//2
    ]
    
    # Lines equations from 2 given points
    cleft = np.poly1d(np.polyfit([left_line[0],left_line[2]],[left_line[1],left_line[3]],1))
    cright = np.poly1d(np.polyfit([right_line[0],right_line[2]],[right_line[1],right_line[3]],1))
    
    if not started:    
        started = 1
        left, right = cleft, cright
    
    else:
        left = alpha*last_left + (1-alpha)*cleft
        right = alpha*last_right + (1-alpha)*cright
    
    # Find top and bottom X points on each line at given heights Y0 Y1
    y0 = img.shape[0]
    y1 = int(height*y0)
    
    l_x0 = int((y0 - left[0])/left[1])
    l_x1 = int((y1 - left[0])/left[1])
    r_x0 = int((y0 - right[0])/right[1])
    r_x1 = int((y1 - right[0])/right[1])
    
    print("center dist: ", (img.shape[1]//2) - (l_x0-r_x0)//2)
    # Draw left and right lines
    cv2.line(img, (l_x0,y0), (l_x1,y1), color, thick)
    cv2.line(img, (r_x0,y0), (r_x1,y1), color, thick)
    
    last_left = left
    last_right = right
    
def detect_image(image):
    # Reading the image source
    img = cv2.imread(image)
    img = cv2.resize(img, (960, 540)) 
    detect_lane(img, alpha = 0.8, opacity = 0)
    
    
    
def detect_lane(img, alpha = 0.2, opacity = 0.7):
    global RED, BLUE, GREEN, LIGHTBLUE
    global last_img
    
    if img is None:
        return last_img
    
    # Convert it to grayscale (1 color channel)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Define suitable yellow color levels
    yellow_low = np.array([20,100,100])
    yellow_high = np.array([30,255,255])
    
    # Create 2 arrays with same image shape, each keeps values in range only
    mask_yellow = cv2.inRange(img, yellow_low, yellow_high)
    mask_white = cv2.inRange(img_gray, 205, 255)
    
    # OR the two masks to keep both values for white  and yellow
    mask = cv2.bitwise_or(mask_yellow, mask_white)
    
    # AND the grayscale image with the final mask
    masked_img = cv2.bitwise_and(img_gray, mask)
    
    # Gaussian blur for some smoothness
    blurry = cv2.GaussianBlur(masked_img, (5,5), 0)
    
    # Detecting edges with Canny Edge Detector
    canny_edges = cv2.Canny(blurry, 10, 15)
    
    roi_img = Region_Of_Interest(canny_edges)
    
    lines = HoughTransform(roi_img)
    
    overlay = img.copy()
    
    if lines is not None:
        sorted_lines = lines_sort(lines)
        draw_lines(img, sorted_lines, BLUE, alpha, height = 0.7, thick = 5)
    else:
        lines = HoughTransform(canny_edges)
        sorted_lines = lines_sort(lines)
        draw_lines(img, sorted_lines, RED, alpha, height = 0.7, thick = 5)
    
    cv2.addWeighted(overlay, opacity, img, 1-opacity, 0, img)
    
    last_img = img
    
    #cv2.imwrite('outimg.jpg', roi_img)

    return img

    #cv2.imshow('test', canny_edges)
    #cv2.imshow('test2', roi_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


#######################


#cap = cv2.VideoCapture("test_videos\challenge.mp4")
cap = cv2.VideoCapture("test_videos\solidWhiteRight.mp4")

while True:
    ret, frame = cap.read()
    cv2.imshow('Video', detect_lane(frame, alpha = 0.8, opacity = 0.3))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()


#detect_image("test_images/lane.jpeg")