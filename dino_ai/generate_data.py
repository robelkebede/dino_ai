import numpy as np
import cv2
import keyboard
from mss import mss
from PIL import ImageGrab

training_data = []

DO_NOTHING = 0
PRESS_UP = 1
PRESS_DOWN=2

mon = {'top':0,'left':0,'width':500,'height':500}

#with mss() as sct:

while True:

    """
    image_mss = np.array(sct.grab(mon))

    im = cv2.cvtColor(image_mss,cv2.COLOR_BGR2GRAY)
    im = cv2.Canny(im,500,500)
    gray = cv2.resize(im,dsize=(50,50))

    cv2.imshow("GRAY ",im)
    """

    
    if keyboard.is_pressed("up"):
        key_pressed = PRESS_UP
        print(key_pressed)
    else:
        key_pressed = DO_NOTHING
        print(key_pressed) 



    #training_data.append([gray,key_pressed])

    #if cv2.waitKey(30) == 27:
        #break


print("savimg training data")
np.save("train_data_mss_v1.npy",training_data)
print("training data SAVED")



