"""
    lanes.py
    Nicholas S. Bradford
    12 Feb 2017

"""

import cv2
import numpy as np
import math

from .config import Constants
from .fit import fitLines


def resizeFrame(img, scale):
    return cv2.resize(img, dsize=None, fx=scale, fy=scale)


def getPerspectivePoints(highres_scale):
    original_width = 1920
    original_height = 1080
    scaled_width = int(highres_scale * original_width)
    scaled_height = int(highres_scale * original_height)
    horizon_height = int(scaled_height / 3.0)
    wing_height = int(scaled_height * 2.0 / 3.0)
    right_width = int(scaled_width * 2.0 / 3.0)
    left_width = int(scaled_width / 3.0)
    topLeft = (int(scaled_width / 2.0 - 50), horizon_height + 50)
    topRight = (int(scaled_width / 2.0 + 50), horizon_height + 50)
    bottomLeft = (left_width, wing_height)
    bottomRight = (right_width, wing_height)
    return topLeft, topRight, bottomLeft, bottomRight


def getPerspectiveMatrixFromPoints(topLeft, topRight, bottomLeft, bottomRight):
    # pts1 = np.float32([[382, 48], [411, 48], [292, 565], [565, 565]])
    # pts2 = np.float32([[0,0],[100,0],[0,1600],[100,1600]])
    pts1 = np.float32([ topLeft, topRight, bottomLeft, bottomRight ])
    pts2 = np.float32([[0,0], [Constants.IMG_SCALED_HEIGHT,0], [0,Constants.IMG_SCALED_HEIGHT], [Constants.IMG_SCALED_HEIGHT,Constants.IMG_SCALED_HEIGHT]])   
    M = cv2.getPerspectiveTransform(pts1,pts2)  
    return M


def getPerspectiveMatrix(highres_scale):
    topLeft, topRight, bottomLeft, bottomRight = getPerspectivePoints(highres_scale)
    perspectiveMatrix = getPerspectiveMatrixFromPoints(topLeft, topRight, bottomLeft, bottomRight)
    return perspectiveMatrix


def addPerspectivePoints(img, topLeft, topRight, bottomLeft, bottomRight):
    cv2.circle(img, topLeft, radius=5, color=(0,0,255))
    cv2.circle(img, topRight, radius=5, color=(0,0,255))
    cv2.circle(img, bottomLeft, radius=5, color=(0,0,255))
    cv2.circle(img, bottomRight, radius=5, color=(0,0,255))


def extractColor(img):
    # green = np.uint8([[[0,255,0 ]]])
    # hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
    # print hsv_green # [[[ 60 255 255]]]
    # yellow: cvScalar(20, 100, 100), cvScalar(30, 255, 255)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower_yellow = np.array([10, 70, 30])
    lower_yellow = np.array([10, 80, 30])
    upper_yellow = np.array([60, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow) # Threshold the HSV image to get only blue colors
    res = cv2.bitwise_and(img, img, mask= mask) # Bitwise-AND mask and original image
    # answer = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    return res


# def extractEdges(img):
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 200, 255, apertureSize=5)
#     bgrEdges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
#     return bgrEdges


def dilateAndErode(img, n_dilations, n_erosions):
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(img, kernel, iterations=n_dilations)
    morphed = cv2.erode(dilated, kernel, iterations=n_erosions)
    return morphed


def skeleton(original):
    # kernel = np.ones((10,10),np.uint8)
    # closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # return closed

    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(original, kernel, iterations=0)
    morphed = cv2.erode(dilated, kernel, iterations=0)

    gray = cv2.cvtColor(morphed, cv2.COLOR_BGR2GRAY)
    size = np.size(gray)
    skel = np.zeros(gray.shape,np.uint8)
    ret,img = cv2.threshold(gray,20,255,0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
    while not done:
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True
    colorSkel = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
    return colorSkel


def addLabels(per, mask, background, colored, dilatedEroded, skeletoned, lines):
    cv2.putText(per, 'Perspective', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
    cv2.putText(mask, 'BackgroundMotionSubtraction', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
    cv2.putText(background, 'Background', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
    cv2.putText(colored, 'Yellow', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
    cv2.putText(dilatedEroded, 'dilated+eroded', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
    cv2.putText(skeletoned, 'skeletoned', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
    # cv2.putText(lines, 'Skeleton+HoughLines', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
    return per, mask, background, colored, dilatedEroded, skeletoned, lines


def show7(img, empty, per, mask, background, colored, lines):
    scale = 0.5
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    empty = cv2.resize(empty, dsize=None, fx=scale, fy=scale)
    per = cv2.resize(per, dsize=None, fx=scale, fy=scale)
    mask = cv2.resize(mask, dsize=None, fx=scale, fy=scale)
    background = cv2.resize(background, dsize=None, fx=scale, fy=scale)
    colored = cv2.resize(colored, dsize=None, fx=scale, fy=scale)
    lines = cv2.resize(lines, dsize=None, fx=scale, fy=scale)

    top = np.hstack((img, per, background))
    bottom = np.hstack((empty, mask, colored, lines))
    cv2.imshow('combined', np.vstack((top, bottom)))


def show9(img, empty, per, mask, background, colored, dilatedEroded, skeletoned, lines):
    scale = 0.5
    img = resizeFrame(img, scale)
    empty = resizeFrame(empty, scale)
    per = resizeFrame(per, scale)
    mask = resizeFrame(mask, scale)
    background = resizeFrame(background, scale)
    colored = resizeFrame(colored, scale)
    lines = resizeFrame(lines, scale)
    dilatedEroded = resizeFrame(dilatedEroded, scale)
    skeletoned = resizeFrame(skeletoned, scale)

    top = np.hstack((img, per, background, mask))
    bottom = np.hstack((empty, colored, dilatedEroded, skeletoned, lines))
    cv2.imshow('combined', np.vstack((top, bottom)))


def laneDetection(img, fgbg, perspectiveMatrix, scaled_height, highres_scale, is_display=True):
    topLeft, topRight, bottomLeft, bottomRight = getPerspectivePoints(highres_scale)
    perspective = cv2.warpPerspective(img, perspectiveMatrix, (scaled_height,scaled_height) )
    fgmask = fgbg.apply(perspective, learningRate=0.5)
    background = fgbg.getBackgroundImage()
    colored = extractColor(background)
    dilatedEroded = dilateAndErode(colored, n_dilations=2, n_erosions=4)
    skeletoned = skeleton(dilatedEroded)
    curve, state = fitLines(skeletoned)
    if is_display:
        addPerspectivePoints(img, topLeft, topRight, bottomLeft, bottomRight)
        per, mask, back, col, dilEroded, skel, lin = addLabels(  perspective, 
                                                cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR), 
                                                background, 
                                                colored, 
                                                dilatedEroded, 
                                                skeletoned,
                                                curve)
        # show7(img, np.zeros((img.shape[0], img.shape[1]-background.shape[1], 3), np.uint8), per, mask, back, col, lin)
        show9(  img, np.zeros((img.shape[0], img.shape[1]-background.shape[1], 3), np.uint8), 
                per, mask, back, col, dilEroded, skel, lin)
    return curve, state

