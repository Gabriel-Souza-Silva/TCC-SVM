#%%
import numpy as np
import os
import pandas as pd

import cv2

import mahotas as mt
import skimage.feature as ftr
import skimage.io

#%%
def extract_features(roi):
    # calculate haralick texture features for 4 types of adjacency
    textures = mt.features.haralick(roi)
    
    # take the mean of it and return it
    ht_mean = textures.mean(axis=0)
    
    return ht_mean


#%%
def findGreatesContour(contours):
    largest_area = 0
    largest_contour_index = -1
    i = 0
    total_contours = len(contours)
    while (i < total_contours ):
        
        area = cv2.contourArea(contours[i])
        
        if(area > largest_area):
            largest_area = area
            largest_contour_index = i
        
        i+=1
            
    return largest_area, largest_contour_index

#%%
def getImageFeatures(path):
    
    image = cv2.imread(path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    bilateralFilter = cv2.bilateralFilter(gray, 9, 100, 100)

    thresh = cv2.threshold(bilateralFilter, 150, 200, cv2.THRESH_BINARY)[1]

    thresh = cv2.erode(thresh, None, iterations=2)

    thresh = cv2.dilate(thresh, None, iterations=2)

    _, contours, hierarchy =  cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0 :
        largest_area, largest_contour_index = findGreatesContour(contours)
        
        x, y, w, h = cv2.boundingRect(contours[largest_contour_index])
        roi = image[y:y+h, x:x+w]
    else:
        # it's need to process non tumour presence so we extract 
        # features from the image itself
        roi = image
    
    grayROI = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    features = extract_features(grayROI)

    return features

#%%
labels = {'yes': 1, 'no':0}
df = pd.DataFrame()
files = []

#%%
for s in ('yes', 'no'):
    path = '/Users/joao/Documents/Projects/TCC_SVM_CLASSIFIER/img/%s' % s

    for file in os.listdir(path):
        if '.jpg' or '.jpeg' in file:
            filename = os.path.join(path, file)

            print("Processing file %s" % filename)
            
            files.append(filename)

            features = getImageFeatures(filename)
            
            row = np.array(features)
            
            row = np.append(row, [labels[s]])

            df = df.append(pd.Series(row), ignore_index=True)


#%%
df.head()


#%%
df.tail()

#%%
df.shape

#%% Implementing model

#defyning data and labels 
X = df.iloc[:, 0:12]
y = df.iloc[:, 13]

#%% splitting dataset into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


#%% working with raw data
from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train, y_train)


#%%
import pickle

filename = './raw_data_trained_model.sav'
pickle.dump(svm, open(filename, 'wb'))


#%%
from sklearn.metrics import accuracy_score

def displayModelPerformance(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


#%% display SVM liner kernel metrics
displayModelPerformance(svm, X_test, y_test)

#%% feature scaling and optimization
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


#%% implementing SVM model
from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)


#%% saving trained model so we don't have to train it everytime
import pickle

# save the model to disk
filename = './optimised_model.sav'
pickle.dump(svm, open(filename, 'wb'))


#%%
# import pickle
# example how to load the model
# loaded_model = pickle.load(open(filename, 'rb'))

#%% display SVM liner kernel metrics
displayModelPerformance(svm, X_test, y_test)


#%% SVM with Radial Basic Function (RBF) kernel for non linear problems
svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
svm.fit(X_train_std, y_train)

# display SVM RBF kernel metrics
displayModelPerformance(svm, X_test_std, y_test)

#%%
#%% applied SVM model for Iris dataset with RBF kernel and higher gamma
svm = SVC(kernel='rbf', random_state=0, gamma=100.0, C=1.0)
svm.fit(X_train_std, y_train)

# display SVM RBF kernel metrics
displayModelPerformance(svm, X_test_std, y_test)

#%%
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)

# display SVM RBF kernel metrics
displayModelPerformance(lr, X_test_std, y_test)

#%%
