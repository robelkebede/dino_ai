import numpy as np
import cv2
import keyboard
from mss import mss
import pyscreenshot as ImageGrab

training_data = []

DO_NOTHING = 0
PRESS_UP = 1
PRESS_DOWN=2

mon = {'top':100,'left':0,'width':500,'height':300}

with mss() as sct:

    while True:

        try:
            key_pressed = 0

            image_mss = np.array(sct.grab(mon))

            im = cv2.cvtColor(image_mss,cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(im,dsize=(50,50))
            gray = cv2.Canny(gray,500,500)

            #print(gray.shape)
            """

            img = ImageGrab.grab(bbox=(90,0,300,200))
            img_np = np.array(img)
            gray = cv2.cvtColor(img_np,cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray,dsize=(50,50)) 
            gray = cv2.Canny(gray,500,500) """
            

            
            if keyboard.is_pressed("up"):
                key_pressed = PRESS_UP
            else:
                key_pressed = DO_NOTHING

            print(key_pressed)

            training_data.append([gray,key_pressed])

            if cv2.waitKey(30) == 27:
                break
        except KeyboardInterrupt as e:
            print("savimg training data")
            np.save("train_data_mss_v8.npy",training_data)
            print("training data SAVED")


