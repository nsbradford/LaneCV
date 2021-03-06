import unittest
import numpy as np

from lanecv.model import MetaModel, MultiModel, LineModel
from lanecv.config import Constants


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

    def test_offsetOrientationToLine(self):
        offset = 1.0 
        offset_pix = LineModel.metersToPixels(offset, pixel_width=Constants.IMG_SCALED_WIDTH, 
                            meters_width=Constants.IMG_WIDTH_IN_METERS)
        orientation_degrees = 270.0 # horizontal
        m, b = LineModel.offsetOrientationToLine(offset_pix, orientation_degrees)
        actual = offset_pix + Constants.IMG_CUTOFF
        print(actual)
        print('DEBUG {} {}'.format(m, b))
        self.assertTrue(np.isclose(0.0, m))
        self.assertTrue(np.isclose(actual, b))


class MetaModelTest(unittest.TestCase):

    def test_chooseBetweenModels_firstIsClosest(self):
        m1 = LineModel(offset=0.4, orientation=170.1)
        m2 = LineModel(offset=0.3, orientation=179.1)
        last_measurement = np.array([0.5, 170.0])
        multimodel = MultiModel(m1, m2)
        choice_multimodel = MetaModel.chooseBetweenModels(multimodel, last_measurement)
        self.assertEquals(choice_multimodel.model1.offset, 0.4)
        self.assertEquals(choice_multimodel.model1.orientation, 170.1)

    def test_chooseBetweenModels_secondIsClosest(self):
        m1 = LineModel(offset=0.3, orientation=179.1)
        m2 = LineModel(offset=0.4, orientation=170.1)
        last_measurement = np.array([0.5, 170.0])
        multimodel = MultiModel(m1, m2)
        choice_multimodel = MetaModel.chooseBetweenModels(multimodel, last_measurement)
        self.assertEquals(choice_multimodel.model1.offset, 0.4)
        self.assertEquals(choice_multimodel.model1.orientation, 170.1)
