import cv2
import numpy as np

hand = cv2.imread('images/hand.jpg',0)

# Finding pixels via thresholding

ret, thre = cv2.threshold(hand, 70, 255, cv2.THRESH_BINARY)

# Contour technique to fuind outer area
_,contours,_ = cv2.findContours(thre.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # returns image, numof contour, heirarchy

# FInding convex hull
hull = [cv2.convexHull(c) for c in contours]

final_output = cv2.drawContours(hand, hull, -1, (255,0,0))


cv2.imshow('Mera photo',hand)

cv2.imshow('Thresh',thre)
cv2.imshow('CONVEX HULL', final_output)

cv2.waitKey(0)
cv2.destroyAllWindows()