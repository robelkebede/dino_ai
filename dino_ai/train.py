import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.svm import SVC
import pandas as pd
import joblib


def old_model():

    dataframe = pd.read_csv("dataset.csv")

    #x = dataframe.drop(["label"],axis=1)
    x = dataframe.drop(["label"],axis=1)
    y = dataframe["label"]


    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=4)

    model = SVC()

    model.fit(x_train,y_train)

    joblib.dump(model,"dino_ai_v2_svc")

    predict = model.predict(x_test)

    print(metrics.classification_report(predict,y_test))

def new_model():
    pass

