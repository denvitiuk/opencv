import cv2
import numpy as np

img_1 = cv2.imread('images/photo1.jpg')
img_2 = cv2.imread('images/photo2.jpg')
img_3 = cv2.imread('images/photo3.jpg')
gray = cv2.cvtColor(img_3, cv2.COLOR_BGR2GRAY)

faces  = cv2.CascadeClassifier('faces.xml')

results = faces.detectMultiScale(gray,scaleFactor=2.3,minNeighbors=2)
for (x,y,w,h) in results:
    cv2.rectangle(img_3,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('Image', img_3)
cv2.waitKey(0)