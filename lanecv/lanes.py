"""
    lanes.py
    12 Feb 2017
    Nicholas S. Bradford

"""

import cv2
import numpy as np
import math

from .config import Constants
from .fit import fitLines
from .plotter import ImageSet, addPerspectivePoints
from .util import getPerspectivePoints


def extractColor(img):
    """ Extract only yellow pixels from img. """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([10, 80, 30]) # orig: (20, 100, 100)
    upper_yellow = np.array([60, 255, 255]) # orig: (30, 255, 255)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow) # Threshold the HSV image to get only blue colors
    res = cv2.bitwise_and(img, img, mask= mask) # Bitwise-AND mask and original image
    return res


def dilateAndErode(img, n_dilations, n_erosions):
    """ Apply morphological dilations then erosions. """
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(img, kernel, iterations=n_dilations)
    morphed = cv2.erode(dilated, kernel, iterations=n_erosions)
    return morphed


def skeleton(original):
    """ Apply morphological skeleton technique to extract 'centers' of thick lines. """
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
    """ Add text labels to pipeline images. """
    cv2.putText(per, 'Perspective', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(mask, 'BackgroundMotionSubtraction', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(background, 'Background', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(colored, 'Yellow', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(dilatedEroded, 'dilated+eroded', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(skeletoned, 'skeletoned', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
    # cv2.putText(lines, 'Skeleton+HoughLines', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
    return per, mask, background, colored, dilatedEroded, skeletoned, lines


def laneDetection(img, fgbg, perspectiveMatrix, scaled_height, highres_scale, is_display=True):
    """ Primary method.
        Args:
            fgbg (BackgroundSubtractorMOG2): background model 
            perspectiveMatrix (np.array): 3x3 matrix for perspective transform
            scaled_height (float): height of the image after scaling
            highres_scale (float): scale for the video frame before processing
            is_display (bool): if True, display particle filters and processing pipeline
        Returns:
            state (MultiModel)
            imgSet (ImageSet)
    """
    topLeft, topRight, bottomLeft, bottomRight = getPerspectivePoints(highres_scale)
    perspective = cv2.warpPerspective(img, perspectiveMatrix, (scaled_height,scaled_height) )
    fgmask = fgbg.apply(perspective, learningRate=0.5)
    background = fgbg.getBackgroundImage()
    colored = extractColor(background)
    dilatedEroded = dilateAndErode(colored, n_dilations=3, n_erosions=3)
    skeletoned = skeleton(dilatedEroded)
    curve, state = fitLines(skeletoned)
    imgSet = None
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
        empty = np.zeros((img.shape[0], img.shape[1]-background.shape[1], 3), np.uint8)
        # show9(  img, empty, 
        #         per, mask, back, col, dilEroded, skel, lin)
        imgSet = ImageSet(img, empty, per, mask, back, col, dilEroded, skel, lin)
    return state, imgSet

