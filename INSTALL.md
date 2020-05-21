# CCTag library

## Building instructions


Required tools:
* CMake >= 3.14 to build the code
* Git
* C/C++ compiler (gcc >= 4.6 or visual studio or clang)

Optional tool:
* CUDA >= 7.0 (CUDA 7.5 is currently not recommended (see Note 1))
Note: On Windows, there are compatibility issues to build the GPU part due to conflicts between msvc/nvcc/thrust/eigen/boost.

### Getting the sources:

```shell
$ git clone https://github.com/alicevision/CCTag.git
```

###  Dependencies

Most of the dependencies can be installed from the common repositories (apt, yum etc):

- Eigen3 (libeigen3-dev)
- Boost >= 1.66 ([accumulators, atomic, chrono, core, date-time, exception, filesystem, math, program-options, ptr-container, system, serialization, stacktrace, timer, thread]-dev)
- OpenCV >= 3.1
- TBB >= 4.0

On a recent Ubuntu-like distribution (e.g. 14.04), you may want to try to run:
```shell
$ sudo apt-get install g++ git-all libpng12-dev libjpeg-dev libeigen3-dev libboost-atomic-dev libboost-chrono-dev libboost-date-time-dev libboost-dev libboost-program-options-dev libboost-exception-dev libboost-filesystem-dev libboost-serialization-dev libboost-system-dev libboost-thread-dev libboost-timer-dev libtbb-dev
```

OpenCV need to be compiled separately and installed in some `OPENCV_INSTALL` path. Then, when running cmake you need to provide the path to the location where `OpenCVConfig.cmake` is installed, usually `${OPENCV_INSTALL}/share/share/OpenCV/` (see below).

CCTag contains code optimized for AVX2  instruction set, which significantly increases detection performance. You can enable it with the option: `cmake -DCCTAG_ENABLE_SIMD_AVX2=ON`.

----------

### Run the building process

You now just need to be in the CCTag folder and run cmake:
```bash
$ mkdir build && cd build
$ cmake .. -DOpenCV_DIR=${OPENCV_INSTALL}/share/share/OpenCV/
$ make -j `nproc`
```

If you want to install the library to, say, a CCTAG_INSTALL path, just add `-DCMAKE_INSTALL_PREFIX=$CCTAG_INSTALL` at cmake command line.
If you want to build CCTag as a shared library: `-DBUILD_SHARED_LIBS=ON`.

----------

### Using CCTag as third party

When you install CCTag a file `CCTagConfig.cmake` is installed in `$CCTAG_INSTALL/lib/cmake/CCTag/` that allows you to import the library in your CMake project.
In your `CMakeLists.txt` file you can add the dependency in this way:

```cmake
# Find the package from the CCTagConfig.cmake
# in <prefix>/lib/cmake/CCTag/. Under the namespace CCTag::
# it exposes the target CCTag that allows you to compile
# and link with the library
find_package(CCTag CONFIG REQUIRED)
...
# suppose you want to try it out in a executable
add_executable(cctagtest yourfile.cpp)
# add link to the library
target_link_libraries(cctagtest PUBLIC CCTag::CCTag)
```

Then, in order to build just pass the location of `CCTagConfig.cmake` from the cmake command line:

```bash
cmake .. -DCCTag_DIR=$CCTAG_INSTALL/lib/cmake/CCTag/
```

## Docker Image

A docker image can be built using the Ubuntu based [Dockerfile](Dockerfile),which is based on nvidia/cuda image (https://hub.docker.com/r/nvidia/cuda/)

A parameter `CUDA_TAG` can be passed when building the image to select the ubuntu and cuda version. 
For example to create a ubuntu 16.04 with cuda 8.0 for development, use
```
docker build --build-arg CUDA_TAG=8.0-devel --tag cctag .
```

The complete list of available tags can be found on the nvidia [dockerhub page](https://hub.docker.com/r/nvidia/cuda/)
In order to run the image nvidia docker is needed: see the installation instruction here https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)
Once installed, the docker can be run, e.g., in interactive mode with

```
docker run -it --runtime=nvidia cctag
```
