"""
    plotter.py
    Nicholas S. Bradsford
    3 April 2017

"""

import cv2
import numpy as np

from .config import Constants
from .particlefilter import ParticleFilterModel
from .model import LineModel


def plotModel(name, img, mymodel, inliers=None, x=None, y=None, color=(255,0,0)):
    # print('RANSAC:, y = {0:.2f}x + {1:.2f} offset {2:.2f} orient {3:.2f}'.format(mymodel.m, mymodel.b, mymodel.offset, mymodel.orientation))
    print('\t{0}: \t*offset {1:.2f} \t orientation {2:.2f}'.format(name, mymodel.offset, mymodel.orientation))
    
    # plot hypothesis
    cv2.line(img=img, pt1=(0,int(mymodel.b)), pt2=(img.shape[1],int(mymodel.m*img.shape[1]+mymodel.b)), 
                        color=color, thickness=2)
    # plot cutoff line
    cv2.line(img=img, pt1=(0, Constants.IMG_CUTOFF), pt2=(Constants.IMG_SCALED_HEIGHT, Constants.IMG_CUTOFF), 
                        color=(255,255,255), thickness=2)
    # for i in range(inliers.size):
    #     xcoord = x[i]
    #     ycoord = y[i]
    #     if inliers[i]:
    #         cv2.circle(img, (xcoord, ycoord), radius=1, color=(0,0,255))
    #     else:
    #         cv2.circle(img, (xcoord, ycoord), radius=1, color=(0,255,0))
    return img


def showFilter(particlefilter, img):
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
        ycoord = int((particlefilter.state_matrix[0] - LineModel.OFFSET_MIN) * transform[0])
        xcoord = int((particlefilter.state_matrix[1] - LineModel.ORIENTATION_MIN) * transform[1])
        cv2.circle(particle_overlay, (xcoord, ycoord), radius=15, color=255) #color=(0,0,255))
    cv2.imshow('particles', particle_overlay)

    if img is not None:
        cv2.imshow('model', plotModel('Filter', img, particlefilter.state, color=(0,0,255)))