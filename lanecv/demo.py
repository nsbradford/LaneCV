"""
    demo.py
    04/5/2017
    Nicholas S. Bradford

"""


import cv2

from .config import Constants
from .lanes import laneDetection
from .model import MetaModel
from .communicate import CommunicationZMQ
from .plotter import showModel, showImgSet
from .util import resizeFrame, getPerspectiveMatrix


def openVideo(filename):
    """ 1920 x 1080 original, 960 x 540 resized """ 
    print('Load video {}...'.format(filename))
    cap = cv2.VideoCapture('./media/' + filename)
    # print('Frame size:', frame.shape)
    return cap


def timerDemo():
    import timeit
    n_iterations = 1
    n_frames = 125
    result = timeit.timeit('runner.particleFilterDemo("intersect.mp4", is_display=False, n_frames={})'.format(n_frames), 
                        setup='import runner;', 
                        number=n_iterations)
    seconds = result / n_iterations
    print('Timing: {} seconds for {} frames of video.'.format(seconds, n_frames))
    print('{} frames / second'.format(n_frames / seconds))


def pictureDemo(path, highres_scale=0.5, scaled_height=Constants.IMG_SCALED_HEIGHT):
    perspectiveMatrix = getPerspectiveMatrix(highres_scale)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    prefix = '../img/taxi/'
    frame = cv2.imread(prefix + path)
    frame = resizeFrame(frame, 0.5)
    img = resizeFrame(frame, highres_scale)
    laneDetection(img, fgbg, perspectiveMatrix, scaled_height, highres_scale)
    cv2.waitKey(0)


def videoDemo(filename, is_display=True, highres_scale=0.5, scaled_height=Constants.IMG_SCALED_HEIGHT, n_frames=-1):
    perspectiveMatrix = getPerspectiveMatrix(highres_scale)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    cap = openVideo(filename)
    count = 0
    while(cap.isOpened()):
        count += 1
        if n_frames > 0 and count > n_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        img = resizeFrame(frame, highres_scale)
        laneDetection(img, fgbg, perspectiveMatrix, scaled_height, highres_scale, is_display=is_display)
        if cv2.waitKey(1) & 0xFF == ord('q'): # 1000 / 29.97 = 33.37
            break
    cap.release()
    cv2.destroyAllWindows()


def getVideoSource(filename):
    cap = openVideo(filename)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            yield frame


def particleFilterDemo(filename, is_display=True, highres_scale=0.5, 
                        scaled_height=Constants.IMG_SCALED_HEIGHT, n_frames=-1):
    """ Video demo with particle filtering applied.
        Args:
            TODO
        Returns:
            None
    """
    metamodel = MetaModel(com=CommunicationZMQ())
    showModel(metamodel, None)
    perspectiveMatrix = getPerspectiveMatrix(highres_scale)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    for i, frame in enumerate(getVideoSource(filename)):
        if n_frames > 0 and i > n_frames:
            break
        if i % 1 != 0:
            continue
        img = resizeFrame(frame, highres_scale)
        state, imgSet = laneDetection(img, fgbg, perspectiveMatrix, scaled_height, 
                            highres_scale, is_display=is_display)
        metamodel.updateState(state)
        metamodel.sendMessage()
        showModel(metamodel, imgSet.lines)
        showImgSet(imgSet)
        if cv2.waitKey(0) & 0xFF == ord('q'): # 1000 / 29.97 = 33.37
            break
    cap.release()
    cv2.destroyAllWindows()

