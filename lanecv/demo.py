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
from .plotter import showModel
from .util import resizeFrame, getPerspectiveMatrix


def pictureDemo(path, highres_scale=0.5, scaled_height=Constants.IMG_SCALED_HEIGHT):
    """ Run laneDetection on a single frame (background subtraction will not work). """
    perspectiveMatrix = getPerspectiveMatrix(highres_scale)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    prefix = '../img/taxi/'
    frame = cv2.imread(prefix + path)
    frame = resizeFrame(frame, 0.5)
    img = resizeFrame(frame, highres_scale)
    laneDetection(img, fgbg, perspectiveMatrix, scaled_height, highres_scale)
    cv2.waitKey(0)


def getVideoSource(filename):
    """ Generator for cleanly yielding frames from a video filename stored in ./media """
    print('Load video {}...'.format(filename))
    cap = cv2.VideoCapture('./media/' + filename)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            yield frame
    cap.release()


def timerDemo():
    """ Times execution of particleFilterDemo() to estimate FPS of processing. """
    import timeit
    n_iterations = 1
    n_frames = 125
    execute_str = 'runner.particleFilterDemo("intersect.mp4", is_display=False, n_frames={})'
    result = timeit.timeit(execute_str.format(n_frames), 
                            setup='import runner;', 
                            number=n_iterations)
    seconds = result / n_iterations
    print('Timing: {} seconds for {} frames of video.'.format(seconds, n_frames))
    print('{} frames / second'.format(n_frames / seconds))


def particleFilterDemo(filename, is_display=True, highres_scale=0.5, 
                        scaled_height=Constants.IMG_SCALED_HEIGHT, n_frames=-1):
    """ Video demo with particle filtering applied.
            Change cv2.waitKey() to cv2.waitKey(0) to automatically pause in-between
            processing each frame (use ESC to advance).
        Args:
            filename (str): relative path to video source file
            is_display (bool): if True, display particle filters and processing pipeline
            highres_scale (float): scale for the video frame before processing
            scaled_height (float): height of the image after scaling
            n_frames (int): num of frames to process from video, or '-1' for all frames.
        Returns:
            None
    """
    metamodel = MetaModel(com=CommunicationZMQ())
    if is_display:
        showModel(metamodel, None)
    perspectiveMatrix = getPerspectiveMatrix(highres_scale)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    for i, frame in enumerate(getVideoSource(filename)):
        print('---------------------\tFrame {}'.format(i+1))
        if n_frames > 0 and i > n_frames:
            break
        if i % 1 != 0:
            continue
        img = resizeFrame(frame, highres_scale)
        state, imgSet = laneDetection(img, fgbg, perspectiveMatrix, scaled_height, 
                            highres_scale, is_display=is_display)
        metamodel.updateState(state)
        metamodel.sendMessage()
        if is_display:
            showModel(metamodel, imgSet.lines)
            imgSet.show()
        if cv2.waitKey(1) & 0xFF == ord('q'): # 1000ms / 29.97 fps = 33.37 s per frame
            break
    cv2.destroyAllWindows()

