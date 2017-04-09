"""
    model.py
    27 March 2017
    Nicholas S. Bradford

"""

import cv2
import numpy as np
import math

from .config import Constants
from .particlefilter import ParticleFilterModel
from .proto import lanecv_pb2


class MetaModel():
    """ A combination of 1 or more ParticleFilterModels. """

    def __init__(self, com):
        # params = {  offset_range=LineModel.OFFSET_RANGE, 
        #             orientation_range=LineModel.ORIENTATION_RANGE, 
        #             offset_min=LineModel.OFFSET_MIN, 
        #             offset_max=LineModel.OFFSET_MAX, 
        #             orientation_min=LineModel.ORIENTATION_MIN, 
        #             orientation_max=LineModel.ORIENTATION_MAX
        #         }
        self.com = com
        self.pfmodel_1 = ParticleFilterModel(particle_cls=LineModel)
        self.pfmodel_2 = ParticleFilterModel(particle_cls=LineModel)


    def updateState(self, multimodel):
        if multimodel.model1 is None and multimodel.model2 is None:
            self.pfmodel_1.updateStateNoEvidence()
            self.pfmodel_2.updateStateNoEvidence()
        elif multimodel.model2 is None:
            self.pfmodel_1.updateState(multimodel.model1)
            self.pfmodel_2.updateStateNoEvidence()
        else:
            multimodel = MetaModel.chooseBetweenModels(multimodel, self.pfmodel_1.state_matrix)
            self.pfmodel_1.updateState(multimodel.model1)
            self.pfmodel_2.updateState(multimodel.model2)
        return MultiModel(self.pfmodel_1.state, self.pfmodel_2.state)


    @staticmethod
    def chooseBetweenModels(multimodel, last_measurement):
        """
            Args:
                multimodel (MultiModel): new evidence
                last_measurement (np.array): previous model
            Returns:
                MultiModel
        """
        m1 = multimodel.model1
        m2 = multimodel.model2
        if multimodel.model2 is not None:
            observations = np.array([   [m1.offset, m1.orientation],
                                        [m2.offset, m2.orientation]])
            distance = ParticleFilterModel._distance(particle_cls=LineModel,
                                new_particles=observations, 
                                measurement=last_measurement)
            print('\t\tChoice 1 dist {0:.2f}: \toffset {1:.2f} \t orientation {2:.2f}'.format(
                                distance[0], m1.offset, m1.orientation))
            print('\t\tChoice 2 dist {0:.2f}: \toffset {1:.2f} \t orientation {2:.2f}'.format(
                                distance[1], m2.offset, m2.orientation))
            if distance[0] > distance[1]:
                m1, m2 = m2, m1
        return MultiModel(m1, m2)


    def sendMessage(self):
        output_file = 'OUTPUT.txt'
        multiMessage = lanecv_pb2.MultiLaneMessage()

        laneMessage1 = multiMessage.laneMessages.add()
        laneMessage1.offset = self.pfmodel_1.state.offset
        laneMessage1.orientation = self.pfmodel_1.state.orientation

        if self.pfmodel_2.state is not None:
            laneMessage2 = multiMessage.laneMessages.add()
            laneMessage2.offset = self.pfmodel_2.state.offset
            laneMessage2.orientation = self.pfmodel_2.state.orientation
        self.com.sendMessage(multiMessage)


    # def writeToOutputFile(multiMessage, output_file)
    #     with open(output_file, 'wb') as f:
    #         f.write(multiMessage.SerializeToString())

    #     mmread = lanecv_pb2.MultiLaneMessage()
    #     with open(output_file, 'rb') as f:
    #         try:
    #             mmread.ParseFromString(f.read())
    #         except IOError:
    #             print('File not found')
    #         print('PROTOBUF: read from file:')
    #         for lm in mmread.laneMessages:
    #             print('\tOffset {} Orientation {}'.format(lm.offset, lm.orientation))


class MultiModel():

    def __init__(self, model1, model2=None):
        self.model1 = model1
        self.model2 = model2



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
        if not m and not b:
            offset_pix = LineModel.pixelsToMeters(self.offset, pixel_width=Constants.IMG_SCALED_WIDTH, 
                            meters_width=Constants.IMG_WIDTH_IN_METERS)
            m, b = LineModel.offsetOrientationToLine(offset_pix, orientation)
        self.m = m
        self.b = b

    @classmethod
    def from_line(cls, m, b):
        offset, orientation = LineModel.lineToOffsetOrientation(m, b)
        return cls(offset, orientation + 180, m=m, b=b)


    @staticmethod
    def modelToMatrix(model):
        """ Convert model to matrix representation. """
        return np.array([model.offset, model.orientation])


    @staticmethod
    def matrixToModel(model_matrix):
        """ Convert the internal matrix to a model. """
        return LineModel(offset=model_matrix[0], orientation=model_matrix[1])


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
    def offsetOrientationToLine(offset_pix, raw_orientation):
        """
            Args:
        """
        angle_offset = 90#- 90 if raw_orientation >= 0 else 90
        orientation = raw_orientation + angle_offset
        m = math.tan(math.radians(orientation))
        b = LineModel.calcIntercept(x0=LineModel.CENTER, y0=LineModel.NOSE_HEIGHT, slope=m, 
                            perp_distance=offset_pix)
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
