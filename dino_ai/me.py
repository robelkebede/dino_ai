from pynput.keyboard import Key, Listener


up,down =1,0
def on_press(key):
    if key is Key.up:
        print("up key pressed ")

def on_release(key):
    pass

# Collect events until released
while True:
    with Listener(on_press=on_press,
            on_release=on_release) as listener:

        listener.join()

