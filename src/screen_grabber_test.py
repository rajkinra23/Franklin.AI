'''
Small unit test, primarily for measuring the load the screen grabber
can handle.

Make sure GTA V is running!
'''

import time
import screen_grabber as sg
from win32 import win32gui

def screen_fps_test(n=100):
    # Initialize start time.
    start = time.time()

    # Grab 250 screnshots.
    for _ in range(n):
        sg.grab_screen()

    # Compute the end time, and in turn the screen fps.
    return float(n)/(time.time() - start)

if __name__ == '__main__':
    # Check that GTA V is running.
    assert win32gui.FindWindow(None, sg.TITLE)

    # Log the screen fps.
    fps = screen_fps_test()
    print("Screen fps: %s" % str(fps))
