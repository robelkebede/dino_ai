import numpy as np
import cv2
import time
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn import metrics
from PIL import Image
import pyautogui
import pyscreenshot as ImageGrab


data = np.load("train_03_resize.npy",allow_pickle=True)
load_model = joblib.load("dino_ai_v5_svc_one_feature")


x = data[:,0]
y = data[:,1]

def trainingVideo():
    for i in range(y.shape[0]):
        time.sleep(0.1)

        cv2.imshow("Image test",x[i])
        print(y[i])

        if cv2.waitKey(30)==27:
            break


def run_game():
    x=1

    while True:
        try:

            img = ImageGrab.grab(bbox=(0,0,500,500))

            img_np = np.array(img)

            gray = cv2.cvtColor(img_np,cv2.COLOR_BGR2GRAY)

            gray = cv2.resize(gray,dsize=(50,50)) 

            cv2.imshow("TEST_#",gray)
            de = predict(gray,load_model)
            print("the de",de)

            if de[0]==1:
                pyautogui.press("up")
                pyautogui.press("enter")
                print("JUMP")
                print([time.ctime(),"JUMP",x])
            elif de[0]==2:
                pyautogui.press("down")
                pyautogui.press("enter")
                print("DOWN")
            else:
                print("DO_NOTHING")
                pyautogui.press("enter")


            x=x+1
            if cv2.waitKey(30)==27:
                break
        except KeyboardInterrupt:
            print("ctrl+c pressed....")
            sys.exit(1)

            break



def contureTest(img):
    img = cv2.GaussianBlur(img,(5,5),2)

    ret,thresh = cv2.threshold(img,10,255,0)
    image, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    features = []

    for i in range(5):
        try:
            area = cv2.contourArea(contours[i])
            features.append(str(area))
        except:
            features.append("0")
    return features

    
def VideoCon():
    for i in range(y.shape[0]):
        time.sleep(0.1)

        contureTest(x[i])

        cv2.imshow("Image test",x[i])
        print(y[i])

        if cv2.waitKey(30)==27:
            break


def predict(img,load_model):
    features = contureTest(img)
    start = time.time()
    predict = load_model.predict(np.array(features[-1]).reshape(1,-1))

    return predict


def main():
    run_game()
   

def trainTest():

    dataframe = pd.read_csv("dataset4.csv")

    x = np.array(dataframe["Area_4"]).reshape(-1,1) 
    y = np.array(dataframe["label"])

    print("xshape ",x.shape)
    print("yshape ",y.shape)
    
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=4)
    
    model = SVC()

    model.fit(x_train,y_train)

    joblib.dump(model,"dino_ai_v5_svc_one_feature")

    predict = model.predict(x_test)

    print(metrics.classification_report(predict,y_test))

if __name__ == "__main__":
    main()


