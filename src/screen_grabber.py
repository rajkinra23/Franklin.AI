from PIL import ImageGrab
from win32 import win32gui
from screengrab_win32 import getRectAsImage
import os

TITLE = "Grand Theft Auto V"
ROOT = 'screens/'

def grab_screen(title=TITLE):
    # Get the GTA V window.
    window = win32gui.FindWindow(None, TITLE)

    # Locate the bounding box of the game screen.
    bbox = win32gui.GetWindowRect(window)

    # Resize the bbox slightly. We seem to get a few excess pixels on the left
    # and right, lets cut those.
    trim_bbox = (bbox[0] + 5, bbox[1], bbox[2] - 5, bbox[3] - 5)

    # Capture the screenshot using the bounding box.
    screen = getRectAsImage(trim_bbox)

    # Return the screen. We could save it/show it for debugging, but
    # that increases the latency tremendously.
    return screen
