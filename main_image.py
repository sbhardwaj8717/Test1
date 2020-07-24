import cv2
import numpy as np

# Task A
imgb=cv2.imread('bird.jpg',1)
#cv2.imshow("bird.jpg",imgb)
print("Image Shape",imgb.shape)
pxb=imgb[213,320]
print(pxb)
imgc=cv2.imread('cat.jpg',1)
print("Image Shape", imgc.shape)
pxc=imgc[195,320]
print(pxc)
imgf=cv2.imread('flowers.jpg',1)
print("Image Shape",imgf.shape)
pxf=imgf[141,320]
print(pxf)
imgh=cv2.imread('horse.jpg',1)
print("Image Shape",imgh.shape)
pxh=imgh[202,320]
print(pxh)

#Task B
cv2.imshow("Original Image",imgc)
cv2.waitKey(0)
B,G,R=cv2.split(imgc)
zeros=np.zeros(imgc.shape[:2],dtype="uint8")
cv2.imshow("cat_red.jpg",cv2.merge([zeros,zeros,R]))
cv2.imwrite("cat_red.jpg",imgc)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Task C
cv2.imshow("Original Image",imgf)
cv2.waitKey(0)
B_f,G_f,R_f=cv2.split(imgf)
B_f=B_f*50
G_f=G_f*50
R_f=R_f*50
imgfa=cv2.cvtColor(imgf, cv2.COLOR_BGR2BGRA)
A_f=np.ones(imgfa.shape[:2],dtype=np.uint8)
imgfa=cv2.merge([B_f,G_f,R_f,A_f])
#print(imgfa.shape)
cv2.imshow("flowers_alpha.png",imgfa)
cv2.imwrite("flowers_alpha.png",imgfa)
cv2.waitKey(0)
cv2.destroyAllWindows()