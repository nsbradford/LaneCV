#!python
"""
    runner.py

"""

import nose
import cv2

from lanecv.lanes import resizeFrame, getPerspectiveMatrix, laneDetection, Constants
from lanecv.particlefilter import ParticleFilterModel
from lanecv.plotter import showFilter


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


def particleFilterDemo(filename, is_display=True, highres_scale=0.5, 
                        scaled_height=Constants.IMG_SCALED_HEIGHT, n_frames=-1):
    """ Video demo with particle filtering applied.
        Args:
            TODO
        Returns:
            None
    """
    model = ParticleFilterModel()
    showFilter(model, None)
    perspectiveMatrix = getPerspectiveMatrix(highres_scale)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    cap = openVideo(filename)
    count = 0
    while(cap.isOpened()):
        count += 1
        # print('Frame #{}'.format(count))
        if n_frames > 0 and count > n_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        if count % 1 != 0:
            continue
        img = resizeFrame(frame, highres_scale)
        curve, state = laneDetection(img, fgbg, perspectiveMatrix, scaled_height, highres_scale, is_display=is_display)
        model.update_state(state)
        showFilter(model, curve)
        if cv2.waitKey(1) & 0xFF == ord('q'): # 1000 / 29.97 = 33.37
            break
    cap.release()
    cv2.destroyAllWindows()


def testAll():
    print('Test...')
    argv = ['fake', 
            '-verbosity=2', 
            '--nocapture', 
            '--with-coverage', 
            '--cover-package=lanecv']
    result = nose.run(argv=argv)
    return result


if __name__ == '__main__':
    testResult = testAll()
    if testResult:
        # pictureDemo('taxi_straight.png')
        # pictureDemo('taxi_side.png')
        # pictureDemo('taxi_curve.png')
        # videoDemo('taxi_intersect.mp4', is_display=True) # framerate of 29.97
        # videoDemo('../../taxi_trim.mp4') # framerate of 29.97
        # timerDemo()
        # particleFilterDemo('../../taxi_trim.mp4')
        particleFilterDemo('taxi_intersect.mp4')
