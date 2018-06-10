CCTag library
===================

----------------------
Building instructions
----------------------

Required tools:
* CMake >= 3.4 to build the code
* git
* C/C++ compiler (gcc >= 4.6 or visual studio or clang)
For CUDA
* CUDA 7.0 (CUDA 7.5 is currently not recommended (see Note 1))

### Getting the sources:
```shell
$ git clone https://github.com/alicevision/CCTag.git
```

###  Dependencies

Most of the dependencies can be installed from the common repositories (apt, yum etc):

- Eigen3 (libeigen3-dev)
- Boost >= 1.53 ([accumulators, atomic, chrono, core, date-time, exception, filesystem, math, program-options, ptr-container, system, serialization, timer, thread]-dev)
- OpenCV >= 3.1
- tbb >= 4.0

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

----------

Note 1: CCTag uses NVidia CUB (CCTag includes a copy of CUB from CUDA 7.0).
Several CUB functions are known to fail with a few NVidia cards including our reference card,
the GTX 980 Ti.
The CUB that is included with CUDA 7.5 does not solve this problem.
