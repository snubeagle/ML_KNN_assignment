#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:20:15 2020

@author: ryan
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import cv2
import os

def load():
    labels = []
    data = []
    
    arr = os.listdir("Data/animals")
    for i in arr:
        labels.append(i)
        
    for i in labels:
        path = os.path.join(os.getcwd(), "Data/animals/", i)
        
        imageList = list(paths.list_images(path))
        
        for j in imageList:
            image = cv2.imread(j)
            image = cv2.resize(image, (32,32), interpolation=cv2.INTER_CUBIC)
            data.append(image)
            
    return (np.array(labels), np.array(data))

def modelTrainer(label, data):
    kVal = [3, 5, 7]
    lVal = [1, 2]
    
    
    Ylabel = []
    for k in range(3000):
        if k <= 1000:
            Ylabel.append(label[0])
        elif (k > 2000):
            Ylabel.append(label[2])
        else:
            Ylabel.append(label[1])
    
    Ylabel = np.array(Ylabel)
    Ylabel = Ylabel.reshape(3000, 1)
    
    (Xtrain, Xtest, Ytrain, Ytest) = train_test_split(data, Ylabel, test_size=0.2, random_state=42)
    (Xtrain, Xval, Ytrain, Yval) = train_test_split(Xtrain, Ytrain, test_size=0.1, random_state=42)
    
    Ytrain = Ytrain.ravel()
    
    for i in kVal:
        for j in lVal:
            print("K value=", i)
            print("L value=", j)
            
            model = KNeighborsClassifier(n_neighbors=i, p=j)
            model.fit(Xtrain, Ytrain)
            print(classification_report(Yval, model.predict(Xval), target_names=le.classes_))
            
    print("Best K=",kVal[2], " Best L measure=",lVal[0])
    model = KNeighborsClassifier(n_neighbors=kVal[2], p = lVal[0])
    model.fit(Xtrain, Ytrain)
    print(classification_report(Ytest, model.predict(Xtest), target_names=le.classes_))
    
    

if __name__=="__main__":
    le = LabelEncoder()
    print("Loading Images")
    
    labels, data = load()
    
    data = data.reshape((data.shape[0], 3072))
    
    labelsle = le.fit_transform(labels)
    
    modelTrainer(labelsle, data)
    