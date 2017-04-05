# LaneCV
Airport taxiway lane detection with OpenCV-Python.

    /lanecv
        lanes.py                laneDetection() and helpers
        fit.py                  fitLines() and helpers
        model.py                MultiModel and LineModel
        particlefilter.py       MetaModel and ParticleFilterModel
        util.py 				Misc utilities, mainly wrappers used across modules
        config.py               Constants such as image size
        plotter.py              Helpful plotting functions
    /test						Unit tests
    /media						Footage for testing
    requirements.txt 			Install with `$ python install -r requirements.txt`
    runner.py


## Usage

Note that you'll need OpenCV compiled with FFMPEG support in order to load videos.

	$ python runner.py

## Overview

A MetaModel is composed of two ParticleFilterModel instances, each of which track a single LineModel. The MetaModel receives updates in the form of a MultiModel (two LineModel instances). 

## TODO

* Build model with prior distribution given Airport model
* Optimize to reduce needless image copying
* Increase dilation and increase resolution
* Fit complex B-snake/spline/curve to yellow-extraction
	* Need to define search space...

## Backlog

* Perspective Transform: widen field, expand upwards to horizon
* Try using ridges/edges instead of color (fails under extreme curves)
* Video Stabilization with Visual Odometry (hard)
