import unittest

from lanes.model import LineModel
from lanes.config import Constants

class LineModelTest(unittest.TestCase):

    # def test_lineToOffsetOrientation_simple(self):
    #     m = 1.0
    #     b = 0.0
    #     offset, orientation = LineModel.lineToOffsetOrientation(m, b)

    def test_perpendicularDistancePixels(self):
        answer = 10
        x0 = LineModel.CENTER
        y0 = LineModel.NOSE_HEIGHT
        m = 0.0
        b = y0 + answer
        dist = LineModel.perpendicularDistancePixels(x0, y0, slope=m, intercept=b)
        self.assertEquals(answer, dist)

    def test_pixelsToMeters(self):
        pixel_offset = Constants.IMG_SCALED_WIDTH / 2
        meters = LineModel.pixelsToMeters(pixel_offset, pixel_width=Constants.IMG_SCALED_WIDTH, 
                            meters_width=Constants.IMG_WIDTH_IN_METERS)
        answer = Constants.IMG_WIDTH_IN_METERS / 2
        self.assertEquals(answer, meters)