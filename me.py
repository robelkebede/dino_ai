import numpy as np
import cv2,time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import joblib


data = np.load("./dataset/train_data_mss_v3.npy",allow_pickle=True)

x = [np.array(cv2.resize(img,(50,50))) for img in data[:,0]]
y = data[:,1]


def trainingVideo():
    #model = joblib.load("dino_neural_v1")
    for i,img in enumerate(x):
        #time.sleep(1.00001)

        cv2.imshow("Image ",img)

        print(y[i])
        #print([model.predict(img),y[i]])

        if cv2.waitKey(30)==27:
            break


trainingVideo()
