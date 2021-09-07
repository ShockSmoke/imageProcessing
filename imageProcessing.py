import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX


while (1):
    
    ret,frame = cap.read()
    b,g,r= cv2.split(frame)

    # applying CLAHE on frames
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equc_b= clahe.apply(b)
    equc_g = clahe.apply(g)
    equc_r = clahe.apply(r)
    equc = cv2.merge((equc_b, equc_g, equc_r))
    frame=equc

    kernel = np.ones((3, 3), np.uint8)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #below values should be corrected according to image
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])


    #creating mask
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.dilate(mask, kernel, iterations=4)
    mask = cv2.GaussianBlur(mask, (5, 5), 100)

    #finding contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:

        areas = [cv2.contourArea(c) for c in contours]
        print(areas)
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        area= cv2.contourArea(cnt)

    # cv2.drawContours(frame, cnt, -1, (0,0,255), 3)

    # approx the contour a little
    epsilon = 0.0005 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    distance = 2*(10**(-7))* (area**2) - (0.0067 * area) + 83.487
    M = cv2.moments(cnt)
    Cx = int(M['m10']/M['m00'])
    Cy = int(M['m01'] / M['m00'])


    S = 'Distance Of Object: ' + str(distance)
    cv2.putText(frame, S, (5, 50), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    # cv2.imshow('res', res)

    cv2.imshow('frame',frame)
    print(area)

    k = cv2.waitKey(5) & 0xFF

    if k == 27:
        break


cv2.destroyAllWindows()

# cap.release()