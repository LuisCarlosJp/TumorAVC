import cv2
import numpy as np
from matplotlib import pyplot as plt


# Kmeans color segmentation
def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))

file_path = "/home/shirou/Downloads/archive(1)/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0/Patients_CT/049/brain/14.jpg"
image = cv2.imread(file_path)

original = image.copy()
kmeans = kmeans_color_quantization(image, clusters=4)

# Convert to grayscale, Gaussian blur, adaptive threshold
gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3), 0)
thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,2)

# Draw largest enclosing circle onto a mask
mask = np.zeros(original.shape[:2], dtype=np.uint8)
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
for c in cnts:
    ((x, y), r) = cv2.minEnclosingCircle(c)
    cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 2)
    cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
    break

# Bitwise-and for result
result = cv2.bitwise_and(original, original, mask=mask)
result[mask==0] = (255,255,255)

result=cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
# global thresholding
ret1,th1 = cv2.threshold(result,127,255,cv2.THRESH_BINARY)

# Otsu's thresholding
ret2,th2 = cv2.threshold(result,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(result,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# # plot all the images and their histograms
# images = [result, 0, th1,
#           result, 0, th2,
#           blur, 0, th3]
# titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
#           'Original Noisy Image','Histogram',"Otsu's Thresholding",
#           'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

# for i in range(0,3):
#     plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
#     plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
#     plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
#     plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
# plt.show()
rows=th3.shape[0]
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                            param1=100, param2=30,
                            minRadius=190, maxRadius=228)


mask= np.zeros_like(result)
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        mask=cv2.circle(mask, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        mask=cv2.circle(mask, center, radius, (255, 255, 255), -1)

result = cv2.bitwise_and(result, result, mask=mask)


# kmeans=cv2.cvtColor(result,cv2.COLOR_GRAY2BGR)
# kmeans=kmeans_color_quantization(kmeans,clusters=3)

# resultado = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(result,240,255,0)
cv2.imshow("detectedad ", result)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

mask= np.zeros_like(result)
mask= cv2.drawContours(mask, contours, -1, (255,255,255), -1)
result = cv2.bitwise_not(result, result, mask=mask)

ret,thresh = cv2.threshold(result,50,200,0)
cv2.imshow("detectedad ", result)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


mask= np.zeros_like(result)
mask= cv2.drawContours(mask, contours, -1, (255,255,255), 3)
result = cv2.bitwise_not(result, result, mask=mask)

mask= np.zeros_like(result)
for i,cnt in enumerate(contours):
    if hierarchy[0][i][2]== -1:
        if cv2.contourArea(cnt)>200:
            cv2.drawContours(mask, [cnt], 0, (255), -1)


cv2.imshow("detected ", result)
cv2.imshow("detected circles", mask)
cv2.waitKey(0)
# cv2.imshow('thresh', thresh)
#cv2.imshow('result', result)
# cv2.imshow('mask', mask)
# cv2.imshow('kmeans', kmeans)
# cv2.imshow('image', image)

#cv2.waitKey()
