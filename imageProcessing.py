import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('test.mp4')
font = cv2.FONT_HERSHEY_COMPLEX


while (1):
    
    frame= cv2.imread('test.jpeg')
    b,g,r= cv2.split(frame)


    bt=24
    gt=42
    rt=222
       
    b_ = b - bt
    g_ = g - gt
    r_ = r + rt





    # applying CLAHE on frames
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equc_b= clahe.apply(b_)
    equc_g = clahe.apply(g_)
    equc_r = clahe.apply(r_)
    equc = cv2.merge((equc_b, equc_g, equc_r))



    kernel = np.ones((3, 3), np.uint8)
    hsv = cv2.cvtColor(equc, cv2.COLOR_BGR2HSV)



    #below values should be corrected according to image
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])



    #creating mask
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.dilate(mask, kernel, iterations=4)
    mask = cv2.GaussianBlur(mask, (5, 5), 100)

    cv2.imshow('mask',mask)

    #finding contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:

        areas = [cv2.contourArea(c) for c in contours]


        #finding max and second max contour
        mx=0
        secondmax=0
        n =len(areas)
        for i in range(2,n):
            if areas[i]>mx:
                secondmax=mx
                mx=i
            elif areas[i]>secondmax and \
                mx != i:
                secondmax=i
        cnt = contours[mx]
        cnt2= contours[secondmax]



    area= cv2.contourArea(cnt)

    # approx the contour a little
    epsilon = 0.0005 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    epsilon = 0.0005 * cv2.arcLength(cnt2, True)
    approx = cv2.approxPolyDP(cnt2, epsilon, True)




    #finding centre
    M = cv2.moments(cnt)
    Cx = int(M['m10']/M['m00'])
    Cy = int(M['m01'] / M['m00'])

    M2 = cv2.moments(cnt2)
    Cx2 = int(M2['m10']/M2['m00'])
    Cy2 = int(M2['m01'] / M2['m00'])

    a= int((Cx + Cx2)/2)
    b= int((Cy + Cy2)/2)

    
    cv2.circle(frame, (a,b), 17, (0, 0, 255), -1)
    
    


    #finding distance
    distance = 2*(10**(-7))* (area**2) - (0.0067 * area) + 83.487
    S = 'Distance Of Object: ' + str(distance)
    cv2.putText(frame, S, (5, 50), font, 2, (0, 0, 255), 2, cv2.LINE_AA)


    #drawing rectangles
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    x,y,w,h = cv2.boundingRect(cnt2)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
 


    cv2.imshow('frame',frame)


    k = cv2.waitKey(5) & 0xFF

    if k == 27:
        break


cv2.destroyAllWindows()

# cap.release()


    
