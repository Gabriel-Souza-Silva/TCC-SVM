import cv2
import numpy as np
import mahotas as mt
import skimage.feature as ftr
import skimage.io

image = cv2.imread('./img/sample2.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

bilateralFilter = cv2.bilateralFilter(gray, 9, 100, 100)

thresh = cv2.threshold(bilateralFilter, 150, 200, cv2.THRESH_BINARY)[1]

thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)

_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def findGreatestContours(contours):
    largest_area = 0
    largest_contour_idx = -1
    i = 0
    total_contours = len(contours)
    while(i < total_contours):
        area = cv2.contourArea(contours[i])
        if(area > largest_area):
            largest_area = area
            largest_contour_idx = 1
        i=1

    return largest_area, largest_contour_idx

largest_area, largest_contour_index = findGreatestContours(contours)

cv2.drawContours(image, contours[largest_contour_index], -1, (0, 0, 255), 3)
resize = cv2.resize(image,None,fx=0.75,fy=0.75)

x, y, w, h = cv2.boundingRect(contours[largest_contour_index]) 

roi = image[y:y+h, x:x+w]
cv2.rectangle(image,(x,y),( x + w, y + h ),(0,255,0),2) 
# cv2.imshow('area',image)
# cv2.waitKey(0)

roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
grayROI = skimage.img_as_ubyte(roi)
grayROI = grayROI / 32
featMatrix = ftr.greycomatrix(grayROI,[1],[0],levels=8,symmetric=False,normed=True)

def extract_features(image):
    # calculate haralick texture features for 4 types of adjacency
    textures = mt.features.haralick(image)
    # take the mean of it and return it
    ht_mean = textures.mean(axis=0)
    return ht_mean

grayROI = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
features = extract_features(grayROI)
train_features = []

train_features.append(features)

print ("Training features: {}".format(np.array(train_features).shape))
print ("Feature 1 : {}".format(train_features[0][0]))
print ("Feature 2 : {}".format(train_features[0][1]))
print ("Feature 3 : {}".format(train_features[0][2]))
print ("Feature 4 : {}".format(train_features[0][3]))
print ("Feature 5 : {}".format(train_features[0][4]))
print ("Feature 6 : {}".format(train_features[0][5]))
print ("Feature 7 : {}".format(train_features[0][6]))
print ("Feature 8 : {}".format(train_features[0][7]))
print ("Feature 9 : {}".format(train_features[0][8]))
print ("Feature 10 : {}".format(train_features[0][9]))
print ("Feature 11 : {}".format(train_features[0][10]))
print ("Feature 12 : {}".format(train_features[0][11]))
print ("Feature 13 : {}".format(train_features[0][12]))

cv2.imshow('final', resize)
cv2.waitKey(0)