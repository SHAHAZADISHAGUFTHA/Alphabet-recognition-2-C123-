import cv2
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
import PIL.ImageOps
import os,ssl,time

X,y = fetcg_openml("mnist_784",vesion = 1,return_X_y = True)
print(pd.Series(y).valuecounts())
classes = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
nclasses = len(classes)

sample_per_class = 5
figure = plt.figure(figsize = (nclasses*2,(1+sample_per_class)))
idx_cls =0 
for cls in classes:
    idxs = np.flatnonzero(y== cls)
    idxs = np.random.choice(idxs,sample_per_class,replace = False)
    i=0
    for idx in idxs:
        plt_idxs = i*nclasses+classes+idx_cls+1
        p = plt.subplot(sample_per_class,nclasses,plt_idxs)
        p = sns.heatmap(np.reshape(X[idx],(28,28)),cmap = plt.cm.gray,xticklabels = Labels,cbar = False)
        p = plt.axis('off')
    idx_cls += 1

x_train,x_test,y_train,y_test = train_test_split(X,y,train_size =7500,test_size= 2500,random_state = 9)
x_trainscaled = x_train/255.0
x_testscaled = x_test/255.0

clf = LogisticRegression(solver="saga", multi_class = "multinomial".fit(x_trainscaled,y_train))

y_pred = clf.predict(x_testscaled)
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

roi = gray[upper_left[1]:bottom_right[1],
upper_left[0]:bottom_right[0]]

image_bw = im_pil.convert('L')
image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)

image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
pixel_filter = 20

min_pixel = np.percentile(image_bw_resized_inverted,pixel_filter)
max_pixel = np.max(image_bw_resized_inverted)

image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel

test_sample = np.array(image_bw_resized_inverted_scaled),reshape(1,784)
test_pred = clf.predict(test_sample)
print("Predicted class:",test_pred)