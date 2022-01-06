#
# script to detect camera movement using orb feature matching
#
# see
# https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/


import numpy as np
import cv2
import argparse
import os

# resolution to display images so they are not cut off
# note that this does not affect the image processing
TARGET_RES = (1280, 720)

# resolution to display two frames side by side
# this does not affect image processing
COMPARE_RES = (1280, 500)

# determines whether or not keypoint matches will be shown for each frame
SHOW_FRAME_POINTS = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script to detect camera\
                                                  movement in cs:go gameplay')

    parser.add_argument('clip', type=str, help='name of gameplay clip in \
                                                ../../gameplay/raw/')
    args = parser.parse_args()

    vidPath = os.path.join('..', '..', 'gameplay', 'raw', args.clip)

    vid = cv2.VideoCapture()
    if not vid.open(vidPath):
        print('failed to open video file')
        exit(1)

    ret, oldFrame = vid.read()
    oldGray = cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY)

    # TODO: use mask to ignore hud

    while True:
        ret, frame = vid.read()

        if not ret:
            print('video ended')
            break

        newGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create()

        # maybe add hud mask like in optical flow script
        oldKeypoints, oldDescriptors = orb.detectAndCompute(oldGray, None)
        newKeypoints, newDescriptors = orb.detectAndCompute(newGray, None)

        matcher = cv2.BFMatcher()
        matches = matcher.match(oldDescriptors,newDescriptors)

        # TODO: remove low quality matches

        oldMatchedPoints = np.zeros((len(matches), 2), dtype=np.float32)
        newMatchedPoints = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            oldMatchedPoints[i,:] = oldKeypoints[match.queryIdx].pt
            newMatchedPoints[i,:] = newKeypoints[match.trainIdx].pt

        hom, mask = cv2.findHomography(oldMatchedPoints, newMatchedPoints,
                                       cv2.RANSAC)

        print(hom)

        if SHOW_FRAME_POINTS:
            img = cv2.drawMatches(oldGray, oldKeypoints, newGray, newKeypoints,
                                  matches, None)
            cv2.imshow('frame', cv2.resize(img, COMPARE_RES))
            cv2.waitKey(0)

        oldGray = newGray.copy()
