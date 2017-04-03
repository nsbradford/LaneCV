"""
    fit.py

"""

import cv2
import numpy as np
from sklearn import linear_model

from .config import Constants
from .model import State, LineModel
from .plotter import plotModel

def extractXY(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    roi = gray[:Constants.IMG_CUTOFF, :]
    y, x = np.nonzero(roi)
    assert x.size == y.size
    return x, y


def isMultiLine(x, y, inliers, outliers):
    debug = True
    m = x.size
    n_outliers = np.count_nonzero(outliers)
    percent_outlier = n_outliers / m
    percent_is_multi = percent_outlier > 0.3
    if percent_outlier < .2:
        if debug: print('isMultiLine(): Too few outliers')
        return False

    combined = np.vstack((x, y))
    cov = np.cov(combined)
    assert cov.shape == (2,2), cov.shape
    evals, evecs = np.linalg.eigh(cov)
    eig_is_multi = evals[1] / 10 < evals[0]

    is_same = eig_is_multi == percent_is_multi
    if debug: print('SameResult: {}\t Eig(cov): {} \t Outlier: {:.1f}%'.format(is_same, eig_is_multi, percent_outlier*100))
    return eig_is_multi or percent_is_multi #, int(evals[0]), int(evals[1])


def fitOneModel(x, y, height, width):
    x = x.reshape(x.size, 1)
    y = y.reshape(y.size, 1)
    # print('\tfit {} {}'.format(x.shape, y.shape))
    model_ransac = linear_model.RANSACRegressor(base_estimator=linear_model.LinearRegression())
                                                #max_trials=1000)
                                                # residual_threshold=5.0 )
    model_ransac.fit(x, y)
    m = model_ransac.estimator_.coef_[0,0]
    b = model_ransac.estimator_.intercept_[0]
    inliers = model_ransac.inlier_mask_
    mymodel = LineModel.from_line(m, b)
    return mymodel, inliers



def fitLines(copy):
    """
        Args:
            img
    """
    img = copy.copy()
    x, y = extractXY(img)
    if x.size < 50:
        print('No lane detected.')
        cv2.putText(img, 'No lane detected', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
        return img, None
    
    try:
        mymodel, inliers = fitOneModel(x, y, height=img.shape[0], width=img.shape[1])
        outliers = ~inliers
        is_multi = isMultiLine(x, y, inliers, outliers)
        img = plotModel('RANSAC-1', img, mymodel, inliers, x, y, color=(255,0,0))
        if is_multi:
            mymodel2, inliers2 = fitOneModel(x[outliers], y[outliers], height=img.shape[0], width=img.shape[1])
            img = plotModel('RANSAC-2', img, mymodel2, inliers2, x, y, color=(0,255,0))
            # if np.count_nonzero(outliers)/inliers.size > .4:
            #     clustering(np.vstack((x, y)).T)
        else:
            mymodel2 = None
    except ValueError as e:
        print('ValueError in model_ransac.fit(): {}'.format(str(e)))
        return img, None
    # print('\tMultiple lines: {}\t{}/{} inliers'.format(is_multi, np.count_nonzero(inliers), inliers.size))
    text = 'offset {0:.2f} orientation {1:.2f}'.format(mymodel.offset, mymodel.orientation)
    # print(text)
    cv2.putText(img, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
    return img, State(mymodel, mymodel2, img)
