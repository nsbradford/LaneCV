
import cv2
import numpy as np
# from scipy.interpolate import UnivariateSpline, CubicSpline
# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
# import scipy.interpolate
from sklearn import linear_model
import math

from config import Constants

from sklearn.feature_extraction import image
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt


def fitRobustLine(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    # line cv2.fitLine(gray, distType=cv2.CV_DIST_L2, param=0, reps, aeps[, line]) 
    # im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    # # then apply fitline() function
    # [vx,vy,x,y] = cv2.fitLine(contours[0] ,cv2.DIST_L2,0,0.01,0.01)

    # # Now find two extreme points on the line to draw line
    # lefty = int((-x*vy/vx) + y)
    # righty = int(((gray.shape[1]-x)*vy/vx)+y)

    # #Finally draw the line
    # cv2.line(img,(gray.shape[1]-1,righty),(0,lefty),255,2)
    # return img

    # coords = cv2.flip(gray, flipCode=0) # flip over x-axis
    y, x = np.nonzero(gray)
    if x.size < 50:
        print('No lane detected.')
        cv2.putText(img, 'No lane detected', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
        return img

    model_ransac = linear_model.RANSACRegressor(base_estimator=linear_model.LinearRegression(),
                                                max_trials=1000)
                                                # residual_threshold=5.0 )
    model_ransac.fit(x.reshape(x.size, 1), y.reshape(x.size, 1))
    m = model_ransac.estimator_.coef_[0,0]
    b = model_ransac.estimator_.intercept_[0]
    mymodel = LineModel(m, b, height=img.shape[0], width=img.shape[1], widthInMeters=3.0)

    print('RANSAC:, y = {0:.2f}x + {1:.2f} offset {2:.2f} orientation {3:.2f}'.format(m, b, mymodel.offset, mymodel.orientation))
    cv2.line(img=img, pt1=(0,int(b)), pt2=(img.shape[1],int(m*img.shape[1]+b)), color=(255,0,0), thickness=2)
    cv2.line(img=img, pt1=(0, Constants.IMG_CUTOFF), pt2=(Constants.IMG_SCALED_HEIGHT, Constants.IMG_CUTOFF), 
                        color=(0,255,0), thickness=2)
    text = 'offset {0:.2f} orientation {1:.2f}'.format(mymodel.offset, mymodel.orientation)
    cv2.putText(img, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
    return img


# def clustering(X):
#     img = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
#     mask = img.astype(bool)

#     img = img.astype(float)
#     img += 1 + 0.2 * np.random.randn(*img.shape)

#     # Convert the image into a graph with the value of the gradient on the
#     # edges.
#     graph = image.img_to_graph(img, mask=mask)

#     # Take a decreasing function of the gradient: we take it weakly
#     # dependent from the gradient the segmentation is close to a voronoi
#     graph.data = np.exp(-graph.data / graph.data.std())

#     Force the solver to be arpack, since amg is numerically
#     unstable on this example
#     labels = spectral_clustering(X, n_clusters=4, eigen_solver='arpack')
#     label_im = -np.ones(mask.shape)
#     label_im[mask] = labels

#     print('{} {} {} '.format(img.shape, graph.shape, label_im.shape))

#     # plt.matshow(img)

#     alg = SpectralClustering(n_clusters=2)
#     print(X.shape)
#     alg.fit(X)
#     plt.figure(1)
#     print('plot...')
#     plt.plot(X[alg.labels_ == 0], color='w')
#     plt.plot(X[alg.labels_ == 1], color='r')
#     plt.show()

# def selectRandomPoints(x_all, y_all):
#     bottom_indices = np.where(y_all == Constants.IMG_CUTOFF)[0]
#     print(bottom_indices.shape)

#     if np.nonzero(bottom_indices)[0].size == 0:
#         return None, None

#     while True:
#         current_bottom = np.random.choice(bottom_indices, size=1)
#         bottom_x = x_all[current_bottom]
#         bottom_y = y_all[current_bottom]
#         print(bottom_x, bottom_y)
#         indices = np.random.randint(low=0, high=x_all.size - 1, size=2)
#         xnew = x_all[indices]
#         ynew = y_all[indices]
#         print(xnew, ynew)
#         x = np.concatenate((bottom_x, xnew))
#         y = np.concatenate((bottom_y, ynew))
#         print(y[0], y[1], y[2])
#         if y[0] < y[1] < y[2]:
#             break;
#     return x, y


# def plotSpline(img):
#     # return
#     print('PlotSpline')
#     # x = np.array([ 2.,  1.,  1.,  2.,  2.,  4.,  4.,  3.])
#     # y = np.array([ 1.,  2.,  3.,  4.,  2.,  3.,  2.,  1.])

#     gray = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), flipCode=0) # flip over x-axis
#     y_all, x_all = np.nonzero(gray)

#     # cv2.imshow('grey', gray)
#     # cv2.waitKey(3000)

#     # for xcoord, ycoord in zip(x, y):
#     #     cv2.circle(img, (xcoord, ycoord), radius=3, color=(0,255,0))
#     # print(x.size)
#     if x_all.size < 50:
#         print('\tWARNING: Not enough points to fit curve')
#         return img
#     plt.axis((0,Constants.IMG_SCALED_HEIGHT,0,Constants.IMG_SCALED_HEIGHT))
    
#     # plt.show()
#     # return

#     x, y = selectRandomPoints(x_all, y_all)
#     if x is None:
#         print('No viable points found.')
#         return
#     plt.scatter(x, y, s=3, marker='o', label='poly')

#     t = np.arange(x.shape[0], dtype=float)
#     t /= t[-1]
#     nt = np.linspace(0, 1, 100)
#     x1 = scipy.interpolate.spline(t, x, nt)
#     y1 = scipy.interpolate.spline(t, y, nt)
#     plt.plot(x1, y1, label='range_spline')

#     t = np.zeros(x.shape)
#     t[1:] = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
#     t = np.cumsum(t)
#     t /= t[-1]
#     x2 = scipy.interpolate.spline(t, x, nt)
#     y2 = scipy.interpolate.spline(t, y, nt)
#     plt.plot(x2, y2, label='dist_spline')

#     plt.legend(loc='best')
#     plt.show()


def fitCurve(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    y, x = np.nonzero(gray)

    # for xcoord, ycoord in zip(x, y):
    #     cv2.circle(img, (xcoord, ycoord), radius=3, color=(0,255,0))
    # print(x.size)
    if x.size < 50:
        print('\tWARNING: Not enough points to fit curve')
        return img

    # TODO x argument must be "strictly" increasing, but this prevents us from handling
    #   splines that are nearly vertical. We'll need a whole new way to fit the curve.
    
    # sortX, sortY = zip(*sorted(zip(x, y)))
    # print(sortX)

    # curve = UnivariateSpline(x=sortX, y=sortY, k=3, s=None) #CubicSpline
    # xs = np.arange(0, img.shape[1], 10)
    # ys = curve(xs)
    # for i in range(xs.size):
    #     if not np.isnan(ys[i]) and 0 < ys[i] < img.shape[0]:
    #         pt = xs[i], int(ys[i])
    #         print(ys[i])
    #         cv2.circle(img, pt, radius=5, color=(0,0,255))
    #         # pt1 = xs[i], int(ys[i])
    #         # pt2 = xs[i + 1], int(ys[i + 1])
    #         # print(pt1, pt2)
    #         # cv2.line(img=img, pt1=pt1, pt2=pt2, color=(0,0,255), thickness=2)
    return img



def addLines(img):
    copy = img.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    minLineLength = 100
    maxLineGap = 10
    # lines = cv2.HoughLinesP(gray,1,np.pi/180,100,minLineLength,maxLineGap)
    # if lines is not None:
    #     for line in lines[:5]:
    #         x1,y1,x2,y2 = line[0]
    #         cv2.line(copy,(x1,y1),(x2,y2),(0,0,255),2)

    # print lines
    # print len(lines)
    # print len(lines[0])
    lines = cv2.HoughLines(image=gray, rho=1, theta=np.pi/180, threshold=100)
    if lines is not None:
        for line in lines: #[0]:
            #print line
            rho,theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)        
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img=copy, pt1=(x1,y1), pt2=(x2,y2), color=(255,0,0), thickness=2)
    cv2.line(img=copy, pt1=(0, Constants.IMG_CUTOFF), pt2=(Constants.IMG_SCALED_HEIGHT, Constants.IMG_CUTOFF), 
                            color=(0,255,0), thickness=2)
    return copy

