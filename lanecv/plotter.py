

import cv2
from .config import Constants

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