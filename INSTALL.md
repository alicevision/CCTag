# CCTag library

## Building instructions

For a detailed guide on building the library check the [online documentation](https://cctag.readthedocs.io/).

Required tools:
* CMake >= 3.14 to build the code
* Git
* C/C++ compiler with C++14 support
    * see here: https://en.cppreference.com/w/cpp/compiler_support 
    * TLDR gcc >= 5, clang >= 3.4, msvc >= 2017

Optional tool:
* CUDA >= 9.0 
Note: On Windows, there are compatibility issues to build the GPU part due to conflicts between msvc/nvcc/thrust/eigen/boost.

###  Dependencies

Most of the dependencies can be installed from the common repositories (apt, yum etc):

- Eigen3 (libeigen3-dev) >= 3.3.4  (NOTE: in order to have Cuda support on Windows, at least version 3.3.9 is required)
- Boost >= 1.66 ([accumulators, atomic, chrono, core, date-time, exception, filesystem, math, program-options, ptr-container, system, serialization, stacktrace, timer, thread]-dev)
- OpenCV >= 3.1
- TBB >= 2021.5.0


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
