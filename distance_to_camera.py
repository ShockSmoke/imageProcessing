from imutils import paths
import numpy as np

import cv2

def find_marker (image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(5,5),0)
    edged=cv2.Canny(gray,50,250,apertureSize=3)
    temp = cv2.dilate(edged, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    cnts,_ = cv2.findContours(temp,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    largestCnt=[]
    for cnt in cnts:
        if (len(cnt)>len(largestCnt)):
            largestCnt=cnt

    return cv2.minAreaRect(cnt)


def distance_to_camera (knownWidth , focalLength ,perWidth):
    return(knownWidth * focalLength)/perWidth

KNOWN_DISTANCE=0.9842
KNOWN_WIDTH=0.09514436
IMAGE_PATHS=["c1.jpeg","c2.jpeg"]
image=cv2.imread(IMAGE_PATHS[0])

marker=find_marker(image)
focalLength = (marker[1][0] * KNOWN_DISTANCE)/KNOWN_WIDTH
cv2.imshow(" test",image)


for imagePath in IMAGE_PATHS:
    image=cv2.imread(imagePath)
    marker=find_marker(image)
    feets=distance_to_camera(KNOWN_WIDTH,focalLength,marker[1][0])

    box=np.int0(cv2.boxPoints(marker))
    cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
    cv2.putText(image, "%0.9842fft" % (feets),
                (image.shape[1] - 200, image.shape[0] -200), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2)
    cv2.imshow("image", image)


cv2.waitKey(0)

cv2.destroyAllWindows()


