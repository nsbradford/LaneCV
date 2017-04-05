"""
	util.py
	4/5/2017
	Nicholas S. Bradford

"""

import numpy as np
import cv2

from .config import Constants


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

