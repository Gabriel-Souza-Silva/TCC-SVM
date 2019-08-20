import mahotas as mt
import numpy as np
import cv2

train_features = []
train_labels = []

def extract_features(image):
    # calculate haralick texture features for 4 types of adjacency
    textures = mt.features.haralick(image)

    # take the mean of it and return it
    ht_mean = textures.mean(axis=0)
    return ht_mean

# read the training image
image = cv2.imread("./img/texture_sample4.jpeg")

# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# extract haralick texture from the image
features = extract_features(gray)

# append the feature vector and label
train_features.append(features)
# train_labels.append(cur_label)

# have a look at the size of our feature vector and labels
print ("Training features: {}".format(np.array(train_features).shape))
print ("Feature 1 : {}".format(train_features[0][0]))
print ("Feature 2 : {}".format(train_features[0][1]))
print ("Feature 3 : {}".format(train_features[0][2]))
print ("Feature 4 : {}".format(train_features[0][3]))
print ("Feature 5 : {}".format(train_features[0][4]))
print ("Feature 6 : {}".format(train_features[0][5]))
print ("Feature 7 : {}".format(train_features[0][6]))
print ("Feature 8 : {}".format(train_features[0][7]))

# print ("Training labels: {}".format(np.array(train_labels).shape))