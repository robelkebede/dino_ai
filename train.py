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

data = np.load("./dataset/train_03_resize.npy",allow_pickle=True)

def model(data):
    x = [cv2.resize(img,(50,50)) for img in data[:,0]]
    y = [tar for tar in data[:,1]]
    
    x = np.array([img for img in x]).reshape(432,50*50)
    y= np.array([tar[0] for tar in y]) 

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=4)

    network = MLPClassifier(hidden_layer_sizes=(100,),verbose=True)

    network.fit(x_train,y_train)

    #joblib.dump(network,"./model/dino_neural_v2")

    predict = network.predict(x_test)

    print(metrics.classification_report(predict,y_test)) 



if __name__ == "__main__":
    model(data)


