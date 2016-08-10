CCTag library
===================

----------------------
Building instructions
----------------------

Required tools:
* CMake to build the code
* git
* C/C++ compiler (gcc >= 4.6 or visual studio or clang)
For CUDA
* CUDA 7.0 (CUDA 7.5 is currently not recommended (see Note 1))

### Getting the sources:
```shell
$ git clone https://github.com/poparteu/CCTag.git
```

###  Dependencies

Most of the dependencies can be installed from the common repositories (apt, yum etc):

- Eigen3 (libeigen3-dev)
- Boost >= 1.53 ([core, thread, system, filesystem, serialization, thread, exception, chrono, date-time, program-options, timer]-dev)
- OpenCV >= 3.1

On a recent Ubuntu-like distribution (e.g. 14.04), you may want to try to run:
```shell
$ sudo apt-get install g++ git-all libpng12-dev libjpeg-dev libeigen3-dev libboost-atomic-dev libboost-chrono-dev libboost-date-time-dev libboost-dev libboost-program-options-dev libboost-exception-dev libboost-filesystem-dev libboost-serialization-dev libboost-system-dev libboost-thread-dev libboost-timer-dev
```

OpenCV need to be compiled separately.

----------

### Run the building process
You now just need to be in the CCTag folder and run cmake:
```bash
$ mkdir build && cd build
$ cmake ..
$ make -j `nproc`
``` 

----------

Note 1: CCTag uses NVidia CUB (CCTag includes a copy of CUB from CUDA 7.0).
Several CUB functions are known to fail with a few NVidia cards including our reference card,
the GTX 980 Ti.
The CUB that is included with CUDA 7.5 does not solve this problem.

