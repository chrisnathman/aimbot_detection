#
# script to detect camera movement using optical flow
#
# see opencv tutorial on optical flow for more explanations of algorithms
# (https://docs.opencv.org/4.5.1/d4/dee/tutorial_optical_flow.html)


import numpy as np
import cv2
import argparse
import os

# resolution to display images so they are not cut off
# note that this does not affect the image processing
TARGET_RES = (1280, 720)

# bins that will be used in histogram step
# pre-generated for efficiency
# bin edges go from -49.5 to 50.5 and increment by 2
HIST_BINS = (np.arange(51) * 2) - 49.5

# scales the size of arrows that are drawn so they are visible
ARROW_SCALAR = 3

# arrow parameters
ARROW_WIDTH = 3
ARROW_TYPE = cv2.LINE_4
ARROW_SHIFT = 0
ARROW_TIP_LENGTH = 0.3

# determines whether or not arrows will be drawn on each frame to show the
# movement of each tracked point
SHOW_POINT_MOVEMENT = True

# color for main screen movement, non random color used here to ensure this
# arrow will always be easily visible
# [255, 0, 255] is magenta
MOVE_COLOR = [255, 0, 255]

# adds arrows for each point tracked
# args - numpy arrays of old and new points being tracked, frame being
#        processed, and array of colors to use to add to the frame
# returns - frame with arrows added for point movement
def show_point_movement(goodOld, goodNew, frame, color):
    ptsMovement = np.subtract(goodNew, goodOld)
    for i,movement in enumerate(ptsMovement):
        startPt = (int(goodOld[i][0]), int(goodOld[i][1]))
        endPt = (startPt[0] + int(round(movement[0], 0))*ARROW_SCALAR,
                 startPt[1] + int(round(movement[1], 0))*ARROW_SCALAR)
        frame = cv2.arrowedLine(frame, startPt, endPt, color[i].tolist(),
                                ARROW_WIDTH, ARROW_TYPE, ARROW_SHIFT,
                                ARROW_TIP_LENGTH)
    return frame


# algorithm to determine screen movement from point movement:
# 1. start with opposite of points movement
# 2. determine which bin each movement x and y value belongs in
# 3. select x and y bins with highest number of values in it
# 4. use median of x and y in the max bins as screen movement x and y
#
# The logic behind this algorithm is that if a lot of points are
# tracked, most points will move based on the movement of the screen.
# Thus errors can be reduced by ignoring movement vectors which are not
# in the most common bin.
#
# This also provides a potential quality indicator which is the ratio
# of elements in the selected bins over the total number of tracked
# points.
#
# args -  numpy arrays of old and new points that were tracked
# returns - (x, y) as integers representing the movement of the screen
def calc_screen_movement(goodOld, goodNew):
    screenMovementVecs = np.subtract(goodOld, goodNew)

    screenMovementXVals = screenMovementVecs[:, 0]
    screenMovementYVals = screenMovementVecs[:, 1]

    # histograms of vector x and y values
    xHistCounts, xHistBins = np.histogram(screenMovementXVals,
                                          bins=HIST_BINS)
    yHistCounts, yHistBins = np.histogram(screenMovementYVals,
                                          bins=HIST_BINS)

    # index of bins with the most elements
    maxXCountIndex = np.argmax(xHistCounts)
    maxYCountIndex = np.argmax(yHistCounts)

    # indices of bins for each movement vector
    xBinIndices = np.digitize(screenMovementXVals, xHistBins)
    yBinIndices = np.digitize(screenMovementYVals, yHistBins)

    # indices of values which are in the largest bins
    # have to add 1 to max count indices because of how digitize() works
    goodXIndices = np.where(xBinIndices == (maxXCountIndex+1))
    goodYIndices = np.where(yBinIndices == (maxYCountIndex+1))

    # x and y values that are in the largest bins
    goodXValues = screenMovementXVals[goodXIndices]
    goodYValues = screenMovementYVals[goodYIndices]

    # final screen movement values
    xMove = np.median(goodXValues)
    yMove = np.median(goodYValues)

    # return movement as integers
    return (int(round(xMove)), int(round(yMove)))


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script to detect camera\
                                                  movement in cs:go gameplay')

    parser.add_argument('clip', type=str, help='name of gameplay clip in \
                                                ../../gameplay/raw/')
    args = parser.parse_args()

    vidPath = os.path.join('..', '..', 'gameplay', 'raw', args.clip)

    color = np.random.randint(0, 255, (100,3))

    vid = cv2.VideoCapture()
    if not vid.open(vidPath):
        print('failed to open video file')
        exit(1)

    ret, prevFrame = vid.read()
    prevGray = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)

    frameHeight, frameWidth, frameChannels = prevFrame.shape

    maskHeight = frameHeight // 4
    maskWidth = frameWidth // 4
    maskP0 = (maskWidth, maskHeight)
    maskP1 = (frameWidth - maskWidth, frameHeight - maskHeight)

    # mask so corner detection will not find points on hud
    hudMask = np.zeros_like(prevGray)

    # masks out 1/4 of frame on top, bottom, left, and right
    # this is seems to be enough to mask out most of the HUD
    hudMask = cv2.rectangle(hudMask, maskP0, maskP1, 255, -1)

    # corner detection parameters from opencv optical flow tutorial
    cornerParams = {'maxCorners': 100, 'qualityLevel': 0.05, 'minDistance': 7,
                    'blockSize': 7}

    # optical flow parameters from opencv optical flow tutorial
    opFlowParams = {'winSize': (15,15), 'maxLevel': 2,
                    'criteria': (cv2.TERM_CRITERIA_EPS |
                                 cv2.TERM_CRITERIA_COUNT, 10, 0.03)}

    oldCorners = cv2.goodFeaturesToTrack(prevGray, mask=hudMask, **cornerParams)

    if len(oldCorners[0][0]) < 1:
        print('no points found to track')
        exit(1)


    while True:
        ret, frame = vid.read()

        if not ret:
            print('video ended')
            break
        
        newGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        newCorners, status, err = cv2.calcOpticalFlowPyrLK(prevGray, newGray,
                                                           oldCorners, None,
                                                           **opFlowParams)

        if newCorners is None:
            print('failed to track points')
            break

        goodNew = newCorners[status==1]
        goodOld = oldCorners[status==1]


        if SHOW_POINT_MOVEMENT:
            frame = show_point_movement(goodOld, goodNew, frame, color)

        screen_movement = calc_screen_movement(goodOld, goodNew)
        frame = show_screen_movement(screen_movement, frame)
        cv2.imshow('frame', cv2.resize(frame, TARGET_RES))
        cv2.waitKey(0)


        prevGray = newGray.copy()

        # redetermine points to track to maintain tracking quality
        oldCorners = cv2.goodFeaturesToTrack(prevGray, mask=hudMask,
                                             **cornerParams)
        if len(oldCorners[0][0]) < 1:
            print('no points found to track')
            exit(1)
