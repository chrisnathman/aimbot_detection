# utility functions for screen movement

import cv2

# scales the size of arrows that are drawn so they are visible
ARROW_SCALAR = 3

# arrow parameters
ARROW_WIDTH = 3
ARROW_TYPE = cv2.LINE_4
ARROW_SHIFT = 0
ARROW_TIP_LENGTH = 0.3

# color for main screen movement, non random color used here to ensure this
# arrow will always be easily visible
# [255, 0, 255] is magenta
MOVE_COLOR = [255, 0, 255]

# adds an arrow representing movement of the screen to the given frame
# args - movement vector (x, y) as integers, frame being processed
# returns - frame with arrow added for screen movement
def show_screen_movement(movement, frame):
    screenCenter = (frame.shape[1]//2, frame.shape[0]//2)
    endPt = (screenCenter[0] + movement[0]*ARROW_SCALAR,
             screenCenter[1] + movement[1]*ARROW_SCALAR)

    frame = cv2.arrowedLine(frame, screenCenter, endPt, MOVE_COLOR, ARROW_WIDTH,
                            ARROW_TYPE, ARROW_SHIFT, ARROW_TIP_LENGTH)
    return frame

