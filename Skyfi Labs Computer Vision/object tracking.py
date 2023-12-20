import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

cam = cv.VideoCapture(0)

lower_yellow = np.array([20,100,100])
upper_yellow = np.array([40,255,255])

while(True):
    #Capture frame by frame 
    ret, frame  = cam.read()
    
    #smoothen image
    image_smooth = cv.GaussianBlur(frame,(7,7),0)

    # threshold the image for yellow colour
    image_hsv = cv.cvtColor(image_smooth, cv.COLOR_BGR2HSV)

    image_threshold = cv.inRange(image_hsv, lower_yellow, upper_yellow)

    #find contours
    image_contour, contours, hierarchy = cv.findContours(image_threshold, \
                                                         cv.RETR_TREE, \
                                                         cv.CHAIN_APPROX_NONE)
    #find the index of the largest contour
    if(len(contours)!=0)
         areas = [cv.contourArea(c)for c in contours]
         max_index = np.argmax(areas)
         cnt = contours[max_index]
         x_bound, y_bound, w_bound, h_bound = cv.boundRect(cnt)
         cv.rectangle(frame, (x_bound, y_bound), (x_bound + w_bound + h_bound),(255,0,0), 2)
         
         
    
    cv.imshow('Frame',frame)
    key = cv.waitKey(10) 
    if key == 27: # wait for escape key to exit 
       break

 img_RGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
 plt.imshow(img_RGB)
 plt.show()
    
    
# when everything done, release the capture 
cam.release() 
cv.destroyAllWindows()




