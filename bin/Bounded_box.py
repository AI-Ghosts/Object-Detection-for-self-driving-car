import numpy as np
import cv2

THRESHOLD = 0.4
VISUILIZATION = True
def update_heatmap(detections, heat):
    for (x1, y1, x2, y2, depth) in detections:
        heat[y1:y2, x1:x2] += depth
    return heat

def get_centroid_rectangles(detections,imgSiz):
    global THRESHOLD
    centroid_rectangles = []

    heat = np.zeros((imgSiz[0], imgSiz[1]), dtype=np.float32) 
    heat = update_heatmap(detections, heat)
    maxValue = np.max(np.abs(heat))
    print(maxValue)
    heat /= maxValue

    _, binary = cv2.threshold(heat.astype(np.float32), THRESHOLD, 255, cv2.THRESH_BINARY)
    if(VISUILIZATION):
        cv2.imshow("Window", heat)
        cv2.waitKey(0)
        cv2.imshow("Window", binary)
        cv2.waitKey(0)

    thresholded_heatmap = binary*maxValue
    thresholded_heatmap = thresholded_heatmap.astype(np.uint8)
    
    _, contours, _ = cv2.findContours(thresholded_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        rect = cv2.boundingRect(contour)
        if rect[2] < 50 or rect[3] < 50: continue
        x, y, w, h = rect
        centroid_rectangles.append([x, y, x + w, y + h])
    return centroid_rectangles