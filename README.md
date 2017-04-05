# LaneCV
Airport taxiway lane detection with OpenCV-Python.

    /lanecv
        lanes.py                laneDetection() and helpers
        fit.py                  fitLines() and helpers
        model.py                MultiModel and LineModel
        particlefilter.py       MetaModel and ParticleFilterModel
        util.py                 Misc utilities, mainly wrappers used across modules
        config.py               Constants such as image size
        plotter.py              Helpful plotting functions
    /test                       Unit tests
    /media                      Footage for testing
    requirements.txt            Install with $ python install -r requirements.txt
    runner.py                   Run a demo.


## Usage

Note that you'll need OpenCV compiled with FFMPEG support in order to load videos.

    $ python runner.py

### Protobuf compilation

After modifying the `lanecv.proto` definition, use `protoc` to recompile the python and java files.

    $ cd ./lanecv/proto
    $ protoc -I=. --python_out=. lanecv.proto
    $ protoc -I=. --java_out=.  lanecv.proto 

## Overview

Initialize by creating a MetaModel, perspectiveMatrix, and backgroundSubtractor. Open the video, and for each frame, update the metamodel state using the results of laneDetection(). A MetaModel is composed of two ParticleFilterModel instances, each of which track a single LineModel. The MetaModel receives updates in the form of a MultiModel (two LineModel instances).

### Assumptions

* Each lane can be approximated as a single line in form [offset, orientation] from the nose of the plane.
* There will never be more than 2 lanes in a single frame (could be changed by adding another step to fitLines() and extending the MetaModel).
* The runway is a flat surface (used for perspective transform).
* The taxiway lane markings are clearly defined yellow.
* The plane motion is reasonably slow (required for background subtraction of the properller, as well as proper particle filtering).

## TODO

### Priorities

* Protobuf 
    * Message architecture
    * Java integration
* Reset ParticleFilterModel after evidence stops being collected
    * This is causing the models to swap positions

### Exploration

* Build model with prior distribution given Airport model
* Fit complex B-snake/spline/curve to yellow-extraction
    * Need to define search space...

###  Backlog

* Optimize to reduce needless image copying
* Increase dilation and increase resolution
* Perspective Transform: widen field, expand upwards to horizon
* Try using ridges/edges instead of color (fails under extreme curves)
* Video Stabilization with Visual Odometry (hard)


## Protobuf steps

* Installed pre-built binary protoc-3.2.0-osx-x86_64.zip from https://github.com/google/protobuf/releases/tag/v3.2.0