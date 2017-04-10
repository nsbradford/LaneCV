#!python
"""
    runner.py

"""

import nose

from lanecv.demo import pictureDemo, videoDemo, timerDemo, particleFilterDemo


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
        timerDemo()
        # particleFilterDemo('../../taxi_trim.mp4')
        # particleFilterDemo('taxi_intersect.mp4')
