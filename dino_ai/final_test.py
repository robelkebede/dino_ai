import pyautogui #for pressing key
import keyboard #for reading key
import time
from threading import Thread
import sys
import cv2
import numpy as np
from train import model
from train import reshape_and_predict
from PIL import ImageGrab

def test_grab():
    while True:
        try:
            img = ImageGrab.grab(bbox=(0,0,500,500))

            img_np = np.array(img)

            gray = cv2.cvtColor(img_np,cv2.COLOR_BGR2GRAY)

            gray = cv2.Canny(gray,100,200)


            cv2.imshow("TEST_#",gray)
        except:
            print("Faild...")

        if cv2.waitKey(30)==27:
            break



def run_game():
    x=1
    while True:
        try:

            #SCREEN GRAB WHILE GAME IS ON
            img = ImageGrab.grab(bbox=(0,0,500,500))

            img_np = np.array(img)

            gray = cv2.cvtColor(img_np,cv2.COLOR_BGR2GRAY)

            gray = cv2.Canny(gray,100,200)


            cv2.imshow("TEST_#",gray)


            #FINAL PREDICTION
            de = reshape_and_predict(gray)
           # de = 0

            if de==1:
                pyautogui.press("up")
                pyautogui.press("enter")
                print("JUMP")
                print([time.ctime(),"JUMP",x])
            elif de==2:
                pyautogui.press("down")
                pyautogui.press("enter")
                print("DOWN")
            else:
                #print([time.ctime(),"DONOT JUMP",x])
                print("DO_NOTHING")
                pyautogui.press("enter")


            x=x+1
            if cv2.waitKey(30)==27:
                break
        except KeyboardInterrupt:
            print("ctrl+c pressed....")
            sys.exit(1)

            break


def main():
     try:
         run_game()
         #test_grab()
     except KeyboardInterrupt:
         print("ctrl +c pressed....")
         sys.exit(1)


if "__main__"== __name__:
        main()
