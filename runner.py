"""
    runner.py
    Nicholas S. Bradford
    May 2017
"""

import nose

from lanecv.demo import pictureDemo, timerDemo, particleFilterDemo


def testAll():
    """ Run all tests. """
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
        # timerDemo()
        particleFilterDemo('taxi_intersect.mp4')
