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

thresh = cv2.threshold(result,150,255,cv2.THRESH_BINARY_INV)[1]

#identification tumor
params = cv2.SimpleBlobDetector_Params()

# Set the threshold
params.minThreshold = 10

# Set the area filter
params.filterByArea = True
params.minArea = 447
params.maxArea = 1450

# Set the circularity filter
params.filterByCircularity = True
params.minCircularity = 0.1


# Set the convexity filter
params.filterByConvexity = True
params.minConvexity = 0.1


# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)
# Detect blobs
keypoints = detector.detect(thresh)

# Draw detected blobs as red circles
img_with_keypoints = cv2.drawKeypoints(original, keypoints, np.array([]), (0, 0, 255),
                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show the image with detected blobs
cv2.imshow("Blobs", img_with_keypoints)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()


