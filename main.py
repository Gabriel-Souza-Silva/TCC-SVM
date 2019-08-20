import cv2
# import numpy as np
# import mahotas as mt
# import skimage.feature as ftr
# import skimage.io

# # def processGetFeature(path): 

# # image = cv2.imread(path)
image = cv2.imread("./img/sample2.png")
cv2.imshow('original >> processed', image)
cv2.waitKey(0)
# # image = cv2.imread("./img/sample3.png")
# # image = cv2.imread("./img/sample4.png")
# # image = cv2.imread("./img/sample5.jpg")
# # image = cv2.imread("./img/sample6.jpg")

# #### processing image

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

'''
Python: cv2.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]])  dst
Parameters:	
    src – input image; the image can have any number of channels, which are processed independently, but the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
    dst – output image of the same size and type as src.
    ksize – Gaussian kernel size. ksize.width and ksize.height can differ but they both must be positive and odd. Or, they can be zero’s and then they are computed from sigma* .
    sigmaX – Gaussian kernel standard deviation in X direction.
    sigmaY – Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to be equal to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height , respectively (see getGaussianKernel() for details); to fully control the result regardless of possible future modifications of all this semantics, it is recommended to specify all of ksize, sigmaX, and sigmaY.
    borderType – pixel extrapolation method (see borderInterpolate for details).
'''
# # gaussianBlurred = cv2.GaussianBlur(gray, (9, 9), cv2.BORDER_DEFAULT) # best result kernel(9x9)

# '''
# Python: cv2.blur(src, ksize[, dst[, anchor[, borderType]]]) → dst
# Parameters:	
#     src – input image; it can have any number of channels, which are processed independently, but the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
#     dst – output image of the same size and type as src.
#     ksize – blurring kernel size.
#     anchor – anchor point; default value Point(-1,-1) means that the anchor is at the kernel center.
#     borderType – border mode used to extrapolate pixels outside of the image.
# '''
# # blur = cv2.blur(gray,(5,5))

# '''
# Python: cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace[, dst[, borderType]]) → dst
# Parameters:	
#     src – Source 8-bit or floating-point, 1-channel or 3-channel image.
#     dst – Destination image of the same size and type as src .
#     d – Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace .
#     sigmaColor – Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood (see sigmaSpace ) will be mixed together, resulting in larger areas of semi-equal color.
#     sigmaSpace – Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ). When d>0 , it specifies the neighborhood size regardless of sigmaSpace . Otherwise, d is proportional to sigmaSpace .
# '''
# bilateralFilter = cv2.bilateralFilter(gray, 9, 100, 100)


# '''
# Python: cv.Threshold(src, dst, threshold, maxValue, thresholdType) → None
# Parameters:	
#     src – input array (single-channel, 8-bit or 32-bit floating point).
#     dst – output array of the same size and type as src.
#     thresh – threshold value.
#     maxval – maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types.
#     type – thresholding type (see the details below).
# '''
# thresh = cv2.threshold(bilateralFilter, 150, 200, cv2.THRESH_BINARY)[1]


# '''
# Python: cv2.erode(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) → dst
# Parameters:	
#     src – input image; the number of channels can be arbitrary, but the depth should be one of CV_8U, CV_16U, CV_16S, CV_32F` or ``CV_64F.
#     dst – output image of the same size and type as src.
#     kernel – structuring element used for erosion; if element=Mat() , a 3 x 3 rectangular structuring element is used. Kernel can be created using getStructuringElement().
#     anchor – position of the anchor within the element; default value (-1, -1) means that the anchor is at the element center.
#     iterations – number of times erosion is applied.
#     borderType – pixel extrapolation method (see borderInterpolate for details).
#     borderValue – border value in case of a constant border
# '''
# thresh = cv2.erode(thresh, None, iterations=2)


# '''
# Python: cv2.dilate(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) → dst
# Parameters:	
#     src – input image; the number of channels can be arbitrary, but the depth should be one of CV_8U, CV_16U, CV_16S, CV_32F` or ``CV_64F.
#     dst – output image of the same size and type as src.
#     kernel – structuring element used for dilation; if elemenat=Mat() , a 3 x 3 rectangular structuring element is used. Kernel can be created using getStructuringElement()
#     anchor – position of the anchor within the element; default value (-1, -1) means that the anchor is at the element center.
#     iterations – number of times dilation is applied.
#     borderType – pixel extrapolation method (see borderInterpolate for details).
#     borderValue – border value in case of a constant border
# '''
# thresh = cv2.dilate(thresh, None, iterations=2)

# # concated = np.concatenate((blurred, thresh), axis=1)
# concated = np.concatenate((gray, thresh), axis=1)
# resize = cv2.resize(concated,None,fx=0.75,fy=0.75)
# # cv2.imshow('original >> processed',resize)
# # cv2.waitKey(0)

# _, contours, hierarchy =  cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# def findGreatesContour(contours):
#     largest_area = 0
#     largest_contour_index = -1
#     i = 0
#     total_contours = len(contours)
#     while (i < total_contours ):
#         area = cv2.contourArea(contours[i])
#         if(area > largest_area):
#             largest_area = area
#             largest_contour_index = i
#         i+=1
            
#     return largest_area, largest_contour_index

# largest_area, largest_contour_index = findGreatesContour(contours)

# # print("largest_area=%.4f and largest_contour_index=%d" % (largest_area, largest_contour_index))

# # cv2.drawContours(image, contours[largest_contour_index], -1, (0, 0, 255), 3)
# # resize = cv2.resize(image,None,fx=0.75,fy=0.75)
# # cv2.imshow('final', resize)
# # cv2.waitKey(0)

# # Get bounding box 
# x, y, w, h = cv2.boundingRect(contours[largest_contour_index]) 

# # Getting ROI 
# roi = image[y:y+h, x:x+w]

# # show ROI 
# # cv2.imshow('segment no',roi) 
# cv2.rectangle(image,(x,y),( x + w, y + h ),(0,255,0),2) 
# # cv2.imshow('area',image)
# # cv2.waitKey(0) 
# # if w > 15 and h > 15: 
# #     cv2.imwrite('C:\\Users\\Link\\Desktop\\output\\{}.png'.format(i), roi)


# # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
# # print(roi)
# # grayROI = skimage.img_as_ubyte(roi)
# # print('================')
# # print(grayROI)
# # grayROI = grayROI / 32
# # print('================')
# # print(grayROI)
# # featMatrix = ftr.greycomatrix(grayROI,[1],[0],levels=8,symmetric=False,normed=True)

# # print (ftr.greycoprops(featMatrix, 'contrast')[0][0])
# # print (ftr.greycoprops(featMatrix, 'energy')[0][0])
# # print (ftr.greycoprops(featMatrix, 'homogeneity')[0][0])
# # print (ftr.greycoprops(featMatrix, 'correlation')[0][0])

# def extract_features(image):
#     # calculate haralick texture features for 4 types of adjacency
#     textures = mt.features.haralick(image)

#     # take the mean of it and return it
#     ht_mean = textures.mean(axis=0)
#     return ht_mean

# grayROI = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# features = extract_features(grayROI)

# train_features = []

# train_features.append(features)

# print ("Training features: {}".format(np.array(train_features).shape))
# print ("Feature 1 : {}".format(train_features[0][0]))
# print ("Feature 2 : {}".format(train_features[0][1]))
# print ("Feature 3 : {}".format(train_features[0][2]))
# print ("Feature 4 : {}".format(train_features[0][3]))
# print ("Feature 5 : {}".format(train_features[0][4]))
# print ("Feature 6 : {}".format(train_features[0][5]))
# print ("Feature 7 : {}".format(train_features[0][6]))
# # print ("Feature 8 : {}".format(train_features[0][7]))
# # print ("Feature 9 : {}".format(train_features[0][8]))
# # print ("Feature 10 : {}".format(train_features[0][9]))
# # print ("Feature 11 : {}".format(train_features[0][10]))
# # print ("Feature 12 : {}".format(train_features[0][11]))
# # print ("Feature 13 : {}".format(train_features[0][12]))
# # return train_features[0]