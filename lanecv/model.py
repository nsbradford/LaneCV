"""
    model.py
    27 March 2017
    Nicholas S. Bradford

"""

import cv2
import numpy as np
import math
from .config import Constants


class State():

    def __init__(self, model1, model2=None, img=None):
        self.model1 = model1
        self.model2 = model2
        self.img = img


class LineModel():
    """ Represents a linear hypothesis for a single lane. 
        Attributes:
            offset (float): shortest (perpendicular) distance to the lane in meters
            orientation (float): that the lane is offset from dead-ahead (+ slope means + degrees)
    """

    OFFSET_MIN = -5.0
    OFFSET_MAX = 5.0
    OFFSET_RANGE = OFFSET_MAX - OFFSET_MIN
    ORIENTATION_MIN = 90.0
    ORIENTATION_MAX = 270.0
    ORIENTATION_RANGE = ORIENTATION_MAX - ORIENTATION_MIN
    CENTER = Constants.IMG_SCALED_WIDTH / 2.0
    NOSE_HEIGHT = Constants.IMG_CUTOFF

    def __init__(self, offset, orientation, m=None, b=None, height=Constants.IMG_SCALED_HEIGHT, 
                        width=Constants.IMG_SCALED_WIDTH, widthInMeters=Constants.IMG_WIDTH_IN_METERS):
        self.offset = offset
        self.orientation = orientation
        offset_pix = LineModel.pixelsToMeters(self.offset, pixel_width=Constants.IMG_SCALED_WIDTH, 
                            meters_width=Constants.IMG_WIDTH_IN_METERS)
        if not m and not b:
            m, b = LineModel.offsetOrientationtoLine(offset, orientation)
        self.m = m
        self.b = b

    @classmethod
    def from_line(cls, m, b):
        offset, orientation = LineModel.lineToOffsetOrientation(m, b)
        return cls(offset, orientation + 180, m=m, b=b)

    @staticmethod
    def lineToOffsetOrientation(m, b):
        """ 
            Args:
                TODO
        """
        pixel_offset = LineModel.perpendicularDistancePixels(x0=LineModel.CENTER, 
                            y0=LineModel.NOSE_HEIGHT, slope=m, intercept=b)
        offset = LineModel.pixelsToMeters(pixel_offset, pixel_width=Constants.IMG_SCALED_WIDTH, 
                            meters_width=Constants.IMG_WIDTH_IN_METERS)
        raw_orientation = math.degrees(math.atan(m))
        angle_offset = - 90 if raw_orientation >= 0 else 90
        orientation = raw_orientation + angle_offset
        return offset, orientation

    @staticmethod
    def perpendicularDistancePixels(x0, y0, slope, intercept):
        """ First, convert [y=mx+b] to [ax+by+c=0]
            f((x0,y0), ax+by+c=0) -> |ax0 + by0 + c| / (a^2 + b^2)^1/2 
        """
        a = slope
        b = -1
        c = intercept
        return abs(a * x0 + b * y0 + c) / math.sqrt(a ** 2 + b ** 2)

    @staticmethod
    def pixelsToMeters(pixel_offset, pixel_width, meters_width):
        """
            Args:
                pixel_offset: offset from lane, in img pixels
                pixel_width: width of image in pixels
                meters_width: width of image in 
        """
        meters_per_pix = meters_width / pixel_width
        return pixel_offset * meters_per_pix

    @staticmethod
    def metersToPixels(meter_offset, pixel_width, meters_width):
        """
            Args:
                pixel_offset: offset from lane, in img pixels
                pixel_width: width of image in pixels
                meters_width: width of image in 
        """
        meters_per_pix = meters_width / pixel_width
        return meter_offset / meters_per_pix

    @staticmethod
    def offsetOrientationtoLine(offset, raw_orientation):
        angle_offset = 90#- 90 if raw_orientation >= 0 else 90
        orientation = raw_orientation + angle_offset
        m = math.tan(math.radians(orientation))
        b = LineModel.calcIntercept(x0=LineModel.CENTER, y0=LineModel.NOSE_HEIGHT, slope=m, 
                            perp_distance=offset)
        return m, b

    @staticmethod
    def calcIntercept(x0, y0, slope, perp_distance):
        first_term = perp_distance * (math.sqrt(slope**2 + (-1)**2))
        second_term = (- slope * x0 + y0)
        sign = LineModel.calcInterceptSign(perp_distance, slope)
        return sign * first_term + second_term

    @staticmethod
    def calcInterceptSign(perp_distance, slope):
        answer = 1
        positive_slope = slope >= 0
        positive_offset = perp_distance >= 0
        if positive_slope and positive_offset:
            answer = -1
        elif positive_slope and not positive_offset:
            answer = 1
        elif not positive_slope and positive_offset:
            answer = 1
        else: # not positive_slope and not positive_offset
            answer = -1
        return answer

