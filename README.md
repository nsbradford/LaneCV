# LaneCV
Airport taxiway lane detection with OpenCV-Python. A part of WPI's [ACAP](http://www.acap.io) (Autonomous Cargo Aircraft Project), completed 2016-2017 by Nicholas Bradford (view the MQP report [here](https://web.wpi.edu/Pubs/E-project/Available/E-project-042717-143558/unrestricted/ACAPFinalReport.pdf)).

![Processing pipeline screenshot](media/results/2screenshot_img.png "Processing pipeline screenshot")

    /lanecv
        demo.py                 Demo code
        lanes.py                laneDetection() and helpers
        fit.py                  fitLines() and helpers
        model.py                MultiModel and LineModel
        particlefilter.py       MetaModel and ParticleFilterModel
        communicate.py          ZMQ messaging
        util.py                 Misc utilities, mainly wrappers used across modules
        config.py               Constants such as image size
        plotter.py              Helpful plotting functions
        /proto                  Protobuf files
            lanecv.proto        Protocol definition file
            lanecv_pb2.py       Python file generated from lanecv.proto
        archive.py              Old code archived away
    /test                       Unit tests
    /media                      Footage for testing
    requirements.txt            Install with $ python install -r 
    runner.py                   Run tests and a demo.


## Usage

### Requirements

Note that you'll need OpenCV compiled with FFMPEG support in order to load videos. Use this [script](https://github.com/nsbradford/ExuberantCV/blob/master/installOpenCV.sh) and some tutorials to understand:

* MacOS
    * The [easy](http://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/) way using Homebrew
    * The [hard](http://www.pyimagesearch.com/2016/11/28/macos-install-opencv-3-and-python-2-7/?__s=6qbo7sdne7fzcniijrik) way
    * With FFMPEG support [here](http://blog.jiashen.me/2014/12/23/build-opencv-3-on-mac-os-x-with-python-3-and-ffmpeg-support/)
* Ubuntu
    * [General](http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/)
    * [With CUDA](http://www.pyimagesearch.com/2016/07/11/compiling-opencv-with-cuda-support/)
    * With ffmpeg support (needed for videos) [here](https://ubuntuforums.org/showthread.php?t=2219550)
    * [Compiling OpenCV with FFMPEG](http://www.wiomax.com/compile-opencv-and-ffmpeg-on-ubuntu/)
* [Installing ffmpeg](http://tipsonubuntu.com/2016/11/02/install-ffmpeg-3-2-via-ppa-ubuntu-16-04/)
* [Compiling ffmpeg from source](http://blog.mycodesite.com/compile-opencv-with-ffmpeg-for-ubuntudebian/)
* Installing CUDA
    * Go [here](https://developer.nvidia.com/cuda-downloads) and download .deb for Ubuntu, DO NOT try the automatic runner it’s a pain 
    * Official documentation [guide](http://developer.download.nvidia.com/compute/cuda/7.5/Prod/docs/sidebar/CUDA_Installation_Guide_Linux.pdf)
    * Helpful blog [post](http://kislayabhi.github.io/Installing_CUDA_with_Ubuntu/)

Then to finish, activate your new `cv` virtual environment and install the other requirements:

    $ workon cv 
    $ pip install -r requirements.txt

### Demo

Run a demo:

    $ python runner.py   

### Protobuf compilation

After modifying the `lanecv.proto` definition, use `protoc` to recompile the python and java files.

    $ cd ./lanecv/proto
    $ protoc -I=. --python_out=. lanecv.proto
    $ protoc -I=. --java_out=.  lanecv.proto 

## Overview

Initialize by creating a MetaModel, perspectiveMatrix, and backgroundSubtractor. Open the video, and for each frame, update the metamodel state using the results of laneDetection(). A MetaModel is composed of two ParticleFilterModel instances, each of which track a single LineModel. The MetaModel receives updates in the form of a MultiModel (two LineModel instances).

### Algorithm

1. Apply perspective transform to Region of Interest (ROI)
2. Update N-frame (2-frame) background model
3. Extract background to eliminate propeller motion
4. Extract yellow color
5. Dilate and erode
6. Apply morphological skeleton procedure to extract “centers” of lanes
7. Predict if there are multiple lanes using covariance matrix eigenvalues
8. Use sequential RANSAC to fit up to two lines to data
9. Update particle filtering models with RANSAC hypotheses
10. Return particle filter estimates

### Assumptions

* Each lane can be approximated as a single line in form [offset, orientation] from the nose of the plane.
* There will never be more than 2 lanes in a single frame (could be changed by adding another step to fitLines() and extending the MetaModel).
* The runway is a flat surface (used for perspective transform).
* The taxiway lane markings are clearly defined yellow.
* The plane motion is reasonably slow (required for background subtraction of the properller, as well as proper particle filtering).

## TODO

### Priorities

* IMPORTANT: Debug offset calculation issues.
* Filtering
    * Make offsets positive and negative
    * Reset ParticleFilterModel after evidence stops being collected
        * This is causing the models to swap positions
        * Use default particle filter settings to tell whether or not lanes is appearing/disappearing- prevents hardcoding of edge cases
* Review skeleton procedure
    * Overview: https://en.wikipedia.org/wiki/Topological_skeleton
    * Our algorithm: https://en.wikipedia.org/wiki/Morphological_skeleton

### Exploration

* Build model with prior distribution given Airport model
* Fit complex B-snake/spline/curve to yellow-extraction
    * Need to define search space...

###  Backlog

* Model particle motion as a moving average of the previous changes; or as plane motion forward; or as plane motion towards the center line.
* Two-way ZMQ-Protobuf integration
* Optimize to reduce needless image copying
* Increase dilation and increase resolution
* Perspective Transform: widen field, expand upwards to horizon
* Try using ridges/edges instead of color (fails under extreme curves)
* Video Stabilization with Visual Odometry (hard)
