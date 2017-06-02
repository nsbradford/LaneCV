"""
    plotter.py
    3 April 2017
    Nicholas S. Bradford

"""

import cv2
import numpy as np

from .config import Constants
from .particlefilter import ParticleFilterModel
from .model import LineModel
from .util import resizeFrame


class ImageSet():

    def __init__(self, img, empty, per, mask, background, colored, dilatedEroded, skeletoned, lines):
        """ Stores the set of processing pipeline images in a standard object. """
        self.img = img
        self.empty = empty
        self.per = per
        self.mask = mask
        self.background = background
        self.colored = colored
        self.dilatedEroded = dilatedEroded
        self.skeletoned = skeletoned
        self.lines = lines

    def show(self):
        """ Shows the image processing pipeline for this frame. """
        scale = 0.5
        img = resizeFrame(self.img, scale)
        empty = resizeFrame(self.empty, scale)
        per = resizeFrame(self.per, scale)
        mask = resizeFrame(self.mask, scale)
        # background = resizeFrame(self.background, scale)
        colored = resizeFrame(self.colored, scale)
        # dilatedEroded = resizeFrame(self.dilatedEroded, scale)
        skeletoned = resizeFrame(self.skeletoned, scale)
        lines = resizeFrame(self.lines, scale)

        top = np.hstack((img, per, mask))
        bottom = np.hstack((empty, colored, skeletoned, lines))
        cv2.imshow('combined', np.vstack((top, bottom)))


def addPerspectivePoints(img, topLeft, topRight, bottomLeft, bottomRight):
    """ Outline in white the border of the perspective projection's region of interest. """
    cv2.circle(img, topLeft, radius=5, color=(255,255,255))
    cv2.circle(img, topRight, radius=5, color=(255,255,255))
    cv2.circle(img, bottomLeft, radius=5, color=(255,255,255))
    cv2.circle(img, bottomRight, radius=5, color=(255,255,255))
    cv2.line(img=img, pt1=topLeft, pt2=topRight, color=(255,255,255), thickness=2)
    cv2.line(img=img, pt1=topLeft, pt2=bottomLeft, color=(255,255,255), thickness=2)
    cv2.line(img=img, pt1=bottomRight, pt2=bottomLeft, color=(255,255,255), thickness=2)
    cv2.line(img=img, pt1=bottomRight, pt2=topRight, color=(255,255,255), thickness=2)


def plotModel(name, img, mymodel, inliers=None, x=None, y=None, color=(255,0,0)):
    """ Add a model visualization to an image. """
    # print('RANSAC:, y = {0:.2f}x + {1:.2f} offset {2:.2f} orient {3:.2f}'.format(
    #                     mymodel.m, mymodel.b, mymodel.offset, mymodel.orientation))
    print('\t{0}: \t*offset {1:.2f} \t orientation {2:.2f}'.format(
                        name, mymodel.offset, mymodel.orientation))
    print('\t{0}: \t*m {1:.2f} \t b {2:.2f}'.format('DEBUG', mymodel.m, mymodel.b))
    # plot hypothesis
    cv2.line(img=img, pt1=(0,int(mymodel.b)), pt2=(img.shape[1],int(mymodel.m*img.shape[1]+mymodel.b)), 
                        color=color, thickness=2)
    # plot cutoff line
    # cv2.line(img=img, pt1=(0, Constants.IMG_CUTOFF), 
    #                     pt2=(Constants.IMG_SCALED_HEIGHT, Constants.IMG_CUTOFF), 
    #                     color=(255,255,255), thickness=2)
    # for i in range(inliers.size):
    #     xcoord = x[i]
    #     ycoord = y[i]
    #     if inliers[i]:
    #         cv2.circle(img, (xcoord, ycoord), radius=1, color=(0,0,255))
    #     else:
    #         cv2.circle(img, (xcoord, ycoord), radius=1, color=(0,255,0))
    return img


def showFilter(text, particlefilter):
    """ Show a grid with white pixels for particles, a big circle showing the current state
            (the avg of all particles), and a small circle at the most recent observation. 
    """
    # print('\tFilter | \t offset {0:.2f} \t orientation {1:.2f}'.format(
    #                     self.state_matrix[0], self.state_matrix[1]))
    length = ParticleFilterModel.VISUALIZATION_SIZE
    img_shape = (length,length)
    shape = (LineModel.OFFSET_RANGE, LineModel.ORIENTATION_RANGE)
    
    particle_overlay = np.zeros(img_shape)
    x = particlefilter.particles + np.array([- LineModel.OFFSET_MIN, - LineModel.ORIENTATION_MIN])
    x = x.clip(np.array([0, 0]), np.array(shape)-1) # Clip out-of-bounds particles
    transform = np.array([length/LineModel.OFFSET_RANGE, length/LineModel.ORIENTATION_RANGE])
    x = (x * transform).astype(int)
    particle_overlay[tuple(x.T)] = 1
    
    if particlefilter.state is not None:
        # show big circle for current hypothesis (avg of all particle positions)
        ycoord = int((particlefilter.state_matrix[0] - LineModel.OFFSET_MIN) * transform[0])
        xcoord = int((particlefilter.state_matrix[1] - LineModel.ORIENTATION_MIN) * transform[1])
        cv2.circle(particle_overlay, (xcoord, ycoord), radius=30, color=255) #color=(0,0,255))

        # show a small circle for the most recent observation
        y_measure = int((particlefilter.last_measurement[0] - LineModel.OFFSET_MIN) * transform[0])
        x_measure = int((particlefilter.last_measurement[1] - LineModel.ORIENTATION_MIN) * transform[1])
        cv2.circle(particle_overlay, (x_measure, y_measure), radius=15, color=128) #color=(0,0,255))
    cv2.imshow(text, particle_overlay)


def showModel(metamodel, img):
    """ Wrapper for plotting the models and particle filters. """
    if metamodel.pfmodel_1 is not None:
        showFilter('model 1', metamodel.pfmodel_1)
    if metamodel.pfmodel_2 is not None:
        showFilter('model 2', metamodel.pfmodel_2)
    if img is not None:
        if metamodel.pfmodel_1.state is not None:
            plotModel('Filter', img, metamodel.pfmodel_1.state, color=(255,0,255))
        if metamodel.pfmodel_2.state is not None:
            plotModel('Filter', img, metamodel.pfmodel_2.state, color=(0,255,255))


    