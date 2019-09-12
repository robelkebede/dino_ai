import numpy as np
import cv2,os
from tqdm import tqdm
import glob
import csv
import time

file = open("dataset4.csv","a")

data = np.load("train_03_resize.npy",allow_pickle=True)

x = data[:,0]
y = data[:,1]

def clean():

    counter = 0
    for img in x:
        img = cv2.GaussianBlur(img,(5,5),2)

        ret,thresh = cv2.threshold(img,5,255,0)
        image, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        features = []

        for i in range(5):
            try:
                area = cv2.contourArea(contours[i])
                features.append(area)
                file.write(str(area))
            except:
                features.append(0)
                file.write("0")

            file.write(",")
            file.write(str(y[counter][0]))
        file.write("\n")

        print([features,y[counter]])
        counter+=1

clean()
