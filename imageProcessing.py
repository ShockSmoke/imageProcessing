import cv2
import numpy as np
from matplotlib import pyplot as plt

# reading image
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_COMPLEX                ##Font style for writing text on video frame






# cv2.namedWindow('track', cv2.WINDOW_NORMAL)


# cv2.createTrackbar('bt','track',0,255, nothing)
# cv2.createTrackbar('gt','track',0,255, nothing)
# cv2.createTrackbar('rt','track',0,255, nothing)



while (1):
#     bt = cv2.getTrackbarPos('bt', 'track')
#     gt = cv2.getTrackbarPos('gt', 'track')
#     rt = cv2.getTrackbarPos('rt', 'track')
     
    #value obtained after trial and run using trackbar (code is commented out)
    bt=24
    gt=42
    rt=222
    
    ret,frame = cap.read()

    b,g,r= cv2.split(frame)
    
    b_ = b - bt
    g_ = g - gt
    r_ = r + rt


    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equc_b= clahe.apply(b_)
    equc_g = clahe.apply(g_)
    equc_r = clahe.apply(r_)
    equc = cv2.merge((equc_b, equc_g, equc_r))

    frame=equc
    kernel = np.ones((3, 3), np.uint8)



    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([110, 50, 50])

    upper_blue = np.array([130, 255, 255])



    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    mask = cv2.dilate(mask, kernel, iterations=4)

    mask = cv2.GaussianBlur(mask, (5, 5), 100)


    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:

        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        area= cv2.contourArea(cnt)

    # cv2.drawContours(frame, cnt, -1, (0,0,255), 3)

    # approx the contour a little
    epsilon = 0.0005 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    #circle around the countour which give as the center of the gate
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(frame, center, radius, (0, 255, 0), 2)
    
    #distance between the camera and the object detected by code
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
    
    cv2.imshow('mask', mask)
    print(area)
    print(center)

    k = cv2.waitKey(5) & 0xFF

    if k == 27:
        break


cv2.destroyAllWindows()
cap.release()
