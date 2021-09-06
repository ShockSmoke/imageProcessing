import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

img = cv2.imread('A1.jpg')

lower_black = np.array([0, 0, 0])
upper_black = np.array([255,255,120])
cv2.namedWindow('image',cv2.WINDOW_NORMAL)


while (True):

    frame=cv2.GaussianBlur(img,(5,5),0)
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,lower_black,upper_black)
    cv2.imshow('mask',mask)
    edges=cv2.Canny(mask,200,250)
    contours,_ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:


        (x,y,w,h) = cv2.boundingRect(contour)
 

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)


        if cv2.waitKey(40)==27:
           break
    cv2.imshow('image',frame)

cv2.destroyAllWindows()
cap.release()