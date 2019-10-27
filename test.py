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
from mss import mss


data = np.load("./dataset/train_03_resize.npy",allow_pickle=True)
load_model = joblib.load("./model/dino_neural_v3")


x = data[:,0]
y = data[:,1]

def trainingVideo():
    for i in range(y.shape[0]):
        time.sleep(0.1)

        cv2.imshow("Image test",x[i])
        print(y[i])

        if cv2.waitKey(30)==27:
            break


def the_copy():

    """
    img = ImageGrab.grab(bbox=(0,0,500,500))

    img_np = np.array(img)

    gray = cv2.cvtColor(img_np,cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray,dsize=(50,50))  

    cv2.imshow("TEST_#",gray)
    """

    """
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

    """



def run_game():
    x=1
    
    with  mss() as sct:
        while True:

            mon = {'top':100,'left':0,'width':500,'height':300}
            image_mss = np.array(sct.grab(mon))

            """

            im = cv2.cvtColor(image_mss, cv2.COLOR_BGR2GRAY) 
            im = cv2.Canny(im,200,200) 
            gray = cv2.resize(im,dsize=(50,50))    """

            im = cv2.cvtColor(image_mss, cv2.COLOR_BGR2GRAY) 

            img = cv2.GaussianBlur(im,(5,5),2)
            #gray = cv2.resize(im,dsize=(50,50),fx=0.5,fy=0.7) 
            canny = cv2.Canny(img,100,200) 
            gray = cv2.resize(canny,dsize=(50,50)) 

            #ret,thresh = cv2.threshold(img,5,255,0)
            """
            img = np.array(ImageGrab.grab(bbox=(0,0,500,500)))

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            gray = cv2.resize(gray,dsize=(50,50))  

            gray = cv2.Canny(gray,150,250)  """


            de = predict(gray,load_model)
            

            if de[0]==1:
                pyautogui.press("up")
                pyautogui.press("enter")
                print([time.ctime(),"JUMP ",de[0]])
            else:
                print([time.ctime(),"WALK"])
                pyautogui.press("enter")


            cv2.imshow("TEST_#",canny)
            
            if cv2.waitKey(30) == 27:
                break



def contureTest(img):

    #img = np.uint8(img)
    img = cv2.GaussianBlur(img,(5,5),2)

    ret,thresh = cv2.threshold(img,5,255,0)
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
        de = predict(x[i],load_model)

        print([y[i],de])

        if cv2.waitKey(30)==27:
            break


def predict(img,load_model):
    #features = np.array(contureTest(img))
    features = img
    predict = load_model.predict(features.reshape(1,-1))
    #predict = load_model.predict(features[-1].reshape(-1,1))
    #check this with more features

    return predict


def main():
    run_game()
   

def trainTest():

    dataframe = pd.read_csv("dataset4.csv")

    x = np.array(dataframe["Area_4"]).reshape(1,-1) 
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
    run_game()
    
    


