import cv2
import numpy as np

img1=cv2.imread('image_1.png')
cv2.imshow("image1.jpg",img1)
print("image shape",img1.shape)

hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

##lower_range_r = np.array([0,50,50])
##upper_range_r = np.array([10,255,255])

# lower mask (0-10)
lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])
mask0 = cv2.inRange(hsv, lower_red, upper_red)

# upper mask (170-180)
lower_red = np.array([170,50,50])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(hsv, lower_red, upper_red)

lower_range_g = np.array([36,65,65])
upper_range_g = np.array([70,255,255])

mask = (mask0 + mask1)

#mask_r = cv2.inRange(hsv, lower_range_r, upper_range_r)
mask_g = cv2.inRange(hsv, lower_range_g, upper_range_g)

output_hsv = hsv.copy()
output_hsv[np.where(mask==0)] = 0

cv2.imshow('image', img1)
#cv2.imshow('mask_r', mask_r)
cv2.imshow('mask_g', mask_g)
cv2.imshow('mask', mask)
cv2.imshow('output', output_hsv)

cv2.waitKey(0)
#cv2.destroyAllWindows()