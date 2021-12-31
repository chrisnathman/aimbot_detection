#
# script to detect camera movement using optical flow
#
# see opencv tutorial on optical flow for more explanations of algorithms
# (https://docs.opencv.org/4.5.1/d4/dee/tutorial_optical_flow.html)
#

import numpy as np
import cv2
import argparse
import os

# resolution to display images so they are not cut off
# note that this does not affect the image processing
TARGET_RES = (1280, 720)

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

    maskHeight = frameHeight / 4
    maskWidth = frameWidth / 4
    maskP0 = (int(maskWidth), int(maskHeight))
    maskP1 = (int(frameWidth - maskWidth), int(frameHeight - maskHeight))

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

    # mask for drawing
    mask = np.zeros_like(prevFrame)

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

        # calculate movement from difference in points
        ptsMovement = np.subtract(goodNew, goodOld)
        screenMovement = np.subtract(goodOld, goodNew)

        ARROW_SCALAR = 3
        for i,movement in enumerate(ptsMovement):
            startPt = (int(goodOld[i][0]), int(goodOld[i][1]))
            endPt = (startPt[0] + int(round(movement[0], 0))*ARROW_SCALAR,
                     startPt[1] + int(round(movement[1], 0))*ARROW_SCALAR)
            frame = cv2.arrowedLine(frame, startPt, endPt, color[i].tolist(), 3,
                                    cv2.LINE_4, 0, 0.3)
            

        #for i,(new,old) in enumerate(zip(goodNew, goodOld)):
        #    a,b = new.ravel()
        #    c,d = old.ravel()
        #    mask = cv2.line(mask, (int(a),int(b)), (int(c),int(d)), color[i].tolist(), 2)
        #    frame = cv2.circle(frame, (int(a),int(b)), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)

        cv2.imshow('frame', cv2.resize(img, TARGET_RES))

        k = cv2.waitKey(0) & 0xff
        if k == 27:
            break

        prevGray = newGray.copy()

        # redetermine points to track to maintain tracking quality
        oldCorners = cv2.goodFeaturesToTrack(prevGray, mask=hudMask,
                                             **cornerParams)
        if len(oldCorners[0][0]) < 1:
            print('no points found to track')
            exit(1)
