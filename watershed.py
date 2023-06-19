import cv2
import numpy as np
from matplotlib import pyplot as plt

#Read file
file_path = "archive(1)/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0/Patients_CT/049/brain/15.jpg"
image = cv2.imread(file_path)

gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
original=gray.copy()

blurImage=cv2.GaussianBlur(gray, [5,5],0)

thresh=cv2.threshold(blurImage, 250,255, cv2.THRESH_BINARY)[1] #threshold the image or change the pixels to make the image easier to analyze (binary: black and white)
dilate=cv2.dilate(thresh, None, iterations=10) #add pixels to the boundaries of the brain (not the inverse of the erosion - produced different image)
erode=cv2.erode(dilate, None, iterations=6) #remove pixels of the boundaries of the brain

#Find second largest contour
contours, hierarchy = cv2.findContours(erode, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)[1]


#Draw mask
aux = np.zeros_like(original, np.uint8)
cv2.drawContours(aux, [sorted_contours], -1, (255,255,255), -1)
cv2.imshow('Mascara', aux)
result = cv2.bitwise_and(original, original, mask=aux)
cv2.imshow('original', result)

#Watershed function
median_filtered = cv2.medianBlur(result, 1)
result=cv2.threshold(median_filtered,155,255,cv2.THRESH_BINARY)[1]
cv2.imshow('Result', result)
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(result,cv2.MORPH_OPEN,kernel, iterations =2)
cv2.imshow('Opening', opening)
sure_bg = cv2.dilate(opening,kernel,iterations=6)
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,0)
ret, sure_fg = cv2.threshold(dist_transform,0.65*dist_transform.max(),255,0)
cv2.imshow('backgroun', sure_bg)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
cv2.imshow('subract', unknown)

ret, marker= cv2.connectedComponents(sure_fg)

marker = marker + 1

marker[unknown==255] = 0

copy_img = image.copy()

cv2.watershed(copy_img, marker)
copy_img[marker==-1]=(0,0,255)
cv2.imshow('Watershed', copy_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
