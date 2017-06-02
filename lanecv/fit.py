"""
    fit.py
    Nicholas S. Bradford
    April 2017

"""

import cv2
import numpy as np
from sklearn import linear_model

from .config import Constants
from .model import MultiModel, LineModel
from .plotter import plotModel


def extractXY(img):
    """ Convert an image into a set of (x, y) coordinates for model fitting. 
        Args:
            img (np.array)
        Returns:
            x (np.array)
            y (np.array)
    """
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    roi = gray[:Constants.IMG_CUTOFF, :]
    y, x = np.nonzero(roi)
    assert x.size == y.size
    return x, y


def isMultiLine(x, y, inliers, outliers):
    """ Returns true if multiple lines are suspected. 
        Begins by returning false immediately if the percentage of outliers
            is less than 30% (to save computation time).
        Then, returns TRUE if the covariance matrix eigenvalues are different
            by an order of magnitude or more.
    """
    debug = True
    m = x.size
    n_outliers = np.count_nonzero(outliers)
    percent_outlier = n_outliers / m
    percent_is_multi = percent_outlier > 0.3
    if percent_outlier < .2:
        if debug: print('isMultiLine(): Too few outliers')
        return False
    else:
        combined = np.vstack((x, y))
        cov = np.cov(combined)
        assert cov.shape == (2,2), cov.shape
        evals, evecs = np.linalg.eigh(cov)
        eig_is_multi = evals[1] / 10 < evals[0]

        is_same = eig_is_multi == percent_is_multi
        if debug: 
            print('SameResult: {}\t Eig(cov): {} \t Outlier: {:.1f}%'.format(
                                is_same, eig_is_multi, percent_outlier*100))
        return eig_is_multi # or percent_is_multi


def fitOneModel(x, y):
    """ Use linear RANSAC to fit a set of points given by their x and y coordinates.
        RANSAC other params to adjust: max_trials=1000, residual_threshold=5.0, ...
        Args:
            x (np.array)
            y (np.array)
        Returns:
            mymodel (LineModel)
            inliers (np.array): mask for inliers determined by the RANSAC model
    """ 
    x = x.reshape(x.size, 1)
    y = y.reshape(y.size, 1)
    model_ransac = linear_model.RANSACRegressor(base_estimator=linear_model.LinearRegression())
    model_ransac.fit(x, y)
    m = model_ransac.estimator_.coef_[0,0]
    b = model_ransac.estimator_.intercept_[0]
    inliers = model_ransac.inlier_mask_
    mymodel = LineModel.from_line(m, b)
    return mymodel, inliers


def fitLines(original_img):
    """ Fit line models to an image.
        Args:
            original_img (np.array)
        Returns:
            img (np.array): image with model visualized on top
            model (MultiModel): contains model (up to 2 lanes fitted)
    """
    img = original_img.copy()
    x, y = extractXY(img)
    if x.size < 50:
        print('No lane detected.')
        cv2.putText(img, 'No lane detected', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                            (0,0,255), 1, cv2.LINE_AA)
        return img, MultiModel(None, None)
    
    try:
        mymodel, inliers = fitOneModel(x, y)
        outliers = ~inliers
        is_multi = isMultiLine(x, y, inliers, outliers)
        plotModel('RANSAC-1', img, mymodel, inliers, x, y, color=(255,0,0))
        if is_multi:
            mymodel2, inliers2 = fitOneModel(x[outliers], y[outliers])
            plotModel('RANSAC-2', img, mymodel2, inliers2, x, y, color=(255,0,0))
        else:
            mymodel2 = None
    except ValueError as e:
        print('ValueError in model_ransac.fit(): {}'.format(str(e)))
        return img, MultiModel(None, None)
    # print('\tMultiple lines: {}\t{}/{} inliers'.format(is_multi, np.count_nonzero(inliers), inliers.size))
    text = 'offset {0:.2f} orientation {1:.2f}'.format(mymodel.offset, mymodel.orientation)
    # print(text)
    cv2.putText(img, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
    return img, MultiModel(mymodel, mymodel2)
