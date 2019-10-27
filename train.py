import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pandas as pd
import joblib


def old_model():
    """
    dataframe = pd.read_csv("dataset_new_4.csv")

    x = dataframe.drop(["label"],axis=1)
    y = dataframe["label"]   """

    data = np.load("./dataset/train_03_resize.npy",allow_pickle=True)

    x = data[:,0]
    y = data[:,1]


    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=4)

    model = SVC()

    model.fit(x_train,y_train)

    joblib.dump(model,"dino_v6_mss")

    predict = model.predict(x_test)

    print(metrics.classification_report(predict,y_test)) 



def new_model():
    data = np.load("./dataset/train_03_resize.npy",allow_pickle=True)
    x = [cv2.resize(img,(50,50)) for img in data[:,0]]
    y = [tar for tar in data[:,1]]
    new_y = []
    """
    
    for i in y:
        if i==1:
            new_y.append([1,0])
        else:
            new_y.append([0,1]) """

    x = np.array([img for img in x]).reshape(432,50*50)
    y= np.array([tar[0] for tar in y]) 

    print(x[0].shape)
    print(y[0])

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=4)


    network = MLPClassifier(verbose=True)

    network.fit(x_train,y_train)

    #joblib.dump(network,"dino_neural_v1")

    predict = network.predict(x_test)

    print(metrics.classification_report(predict,y_test)) 

new_model()
