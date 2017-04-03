import unittest
import numpy as np

from lanecv.model import MultiModel, LineModel
from lanecv.config import Constants
from lanecv.particlefilter import MetaModel


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


class MetaModelTest(unittest.TestCase):

    def test_chooseBetweenModels(self):
        m1 = LineModel(offset=0.4, orientation=170.1)
        m2 = LineModel(offset=0.3, orientation=179.1)
        last_measurement = np.array([0.5, 170.0])
        multimodel = MultiModel(m1, m2)
        choice_multimodel = MetaModel.chooseBetweenModels(multimodel, last_measurement)
        self.assertEquals(choice_multimodel.model1.offset, 0.4)
        self.assertEquals(choice_multimodel.model1.orientation, 170.1)