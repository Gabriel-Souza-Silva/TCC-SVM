# encoding=utf8
import os
import pandas as pd
import main as main

path = './img/'

files = []
imagesFeatures = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.jpg' or '.jpeg' in file:
             files.append(os.path.join(r, file))
            #  features = main.processGetFeature(os.path.join(r, file))
            #  imagesFeatures.append(features)

for f in files:
     print(f)

# print(imagesFeatures[0][0])
# print(imagesFeatures[5][13])

# pd.DataFrame(imagesFeatures).to_csv("output.csv")