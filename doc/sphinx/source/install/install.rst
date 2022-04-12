Requirements
============

Hardware
~~~~~~~~

CCTag has a CPU and a GPU implementation.
The GPU implementation requires an NVIDIA GPU card with a CUDA compute capability >= 3.5.
You can check your `NVIDIA GPU card CC support here <https://github.com/tpruvot/ccminer/wiki/Compatibility>`_ or on the `NVIDIA dev page <https://developer.nvidia.com/cuda-gpus>`_.
If you do not have a NVIDIA card you will still able to compile and use the CPU version.

Here are the minimum hardware requirements for CCTag:

+--------------------------------------------------------------------------+
| Minimum requirements                                                     |
+===================+======================================================+
| Operating systems | Windows x64, Linux, macOS                            |
+-------------------+------------------------------------------------------+
| CPU               | Recent Intel or AMD cpus                             |
+-------------------+------------------------------------------------------+
| RAM Memory        | 8 GB                                                 |
+-------------------+------------------------------------------------------+
| Hard Drive        | No particular requirements                           |
+-------------------+------------------------------------------------------+
| GPU               | NVIDIA CUDA-enabled GPU (compute capability >= 3.5)  |
+-------------------+------------------------------------------------------+



Software
~~~~~~~~

CCTag depends on the following libraries:

* Eigen3  >= 3.3.4

* Boost >= 1.66

* OpenCV >= 3.1

* TBB >= 4.0

.. warning::

   In order to have Cuda support on Windows, at least Eigen 3.3.9 is required


------------


vcpkg
=====

`vcpkg <https://github.com/microsoft/vcpkg>`_ is a cross-platform (Windows, Linux and MacOS), open-source package manager created by Microsoft.

Since v1.0.0 of the library it is possible to build and install the library through vcpkg on Linux, Windows and MacOS by running:

.. code:: shell

  vcpkg install cctag[cuda,apps]

where :code:`cuda` and :code:`apps` are the options to build the library with the cuda support and the sample applications, respectively.

------------

Building the library
====================

Building tools
~~~~~~~~~~~~~~

Required tools:

* CMake >= 3.14 to build the code
* Git
* C/C++ compiler supporting the C++14 standard (gcc >= 5, clang >= 3.4, msvc >= 2017)

Optional tool:

* CUDA >= 9.0


.. note::

  On Windows, there are compatibility issues to build the GPU part due to conflicts between msvc/nvcc/thrust/eigen/boost.


Dependencies
~~~~~~~~~~~~

vcpkg
+++++

vcpkg can be used to install all the dependencies on all the supported platforms.
This is particularly useful on Windows.
To install the dependencies:

.. code:: shell

  vcpkg install
          boost-accumulators
          boost-algorithm
          boost-container
          boost-date-time
          boost-exception
          boost-filesystem
          boost-iterator
          boost-lexical-cast
          boost-math
          boost-mpl
          boost-multi-array
          boost-ptr-container
          boost-program-options
          boost-serialization
          boost-spirit
          boost-static-assert
          boost-stacktrace
          boost-test
          boost-thread
          boost-throw-exception
          boost-timer
          boost-type-traits
          boost-unordered
          opencv
          tbb
          eigen3

You can add the flag :code:`--triplet` to specify the architecture and the version you want to build.
For example:

* :code:`--triplet x64-windows` will build the dynamic version for Windows 64 bit

* :code:`--triplet x64-windows-static` will build the static version for Windows 64 bit

* :code:`--triplet x64-linux-dynamic` will build the dynamic version for Linux 64 bit

and so on.
More information can be found `here <https://vcpkg.readthedocs.io/en/latest/examples/overlay-triplets-linux-dynamic>`_

Linux
+++++

On Linux you can install from the package manager:

For Ubuntu/Debian package system:

.. code:: shell

    sudo apt-get install g++ git-all libpng12-dev libjpeg-dev libeigen3-dev libboost-all-dev libtbb-dev


For CentOS package system:

.. code:: shell

    sudo yum install gcc-c++ git libpng-devel libjpeg-turbo-devel eigen3-devel boost-devel 	tbb-devel


MacOS
+++++

On MacOs using `Homebrew <https://brew.sh/>`_ install the following packages:

.. code:: shell

    brew install git libpng libjpeg eigen boost tbb


Getting the sources
~~~~~~~~~~~~~~~~~~~~

.. code:: shell

   git clone https://github.com/alicevision/CCTag.git


CMake configuration
~~~~~~~~~~~~~~~~~~~

From CCTag root folder you can run cmake:

.. code:: shell

    mkdir build && cd build
    cmake ..
    make -j `nproc`

On Windows add :code:`-G "Visual Studio 16 2019" -A x64` to generate the Visual Studio solution according to your VS version (`see CMake documentation <https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html#ide-build-tool-generators>`_).

If you are using the dependencies built with VCPKG you need to pass :code:`-DCMAKE_TOOLCHAIN_FILE=path/to/vcpkg/scripts/buildsystems/vcpkg.cmake` at cmake step to let it know where to find the dependencies.

Otherwise you can specify the path where each dependency can be found (if not installed in system folders) by passing its related path.
For example, for OpenCV you can pass :code:`-DOpenCV_DIR=path/to/opencv/install/share/OpenCV/` to tell where the :code:`OpenCVConfig.cmake` file can be found.

CMake options
+++++++++++++

CMake configuration can be controlled by changing the values of the following variables (here with their default value)

* :code:`CCTAG_WITH_CUDA:BOOL=ON` to enable/disable the Cuda implementation

* :code:`BUILD_SHARED_LIBS:BOOL=ON` to enable/disable the building shared libraries

* :code:`CCTAG_ENABLE_SIMD_AVX2:BOOL=OFF` to enable/disable the AVX2 optimizations

* :code:`CCTAG_BUILD_TESTS:BOOL=OFF` to enable/disable the building of the unit tests

* :code:`CCTAG_BUILD_APPS:BOOL=ON` to enable/disable the building of applications

* :code:`CCTAG_BUILD_DOC:BOOL=OFF` to enable/disable building this documentation

So if you do not want to build the Cuda part, you have to pass :code:`-DCCTAG_WITH_CUDA:BOOL=OFF` and so on.


------------


CCTag as third party
====================

When you install CCTag a file :code:`CCTagConfig.cmake` is installed in :code:`<install_prefix>/lib/cmake/CCTag/` that allows you to import the library in your CMake project.
In your :code:`CMakeLists.txt` file you can add the dependency in this way:

.. code-block::
  :linenos:

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

Then, in order to build just pass the location of :code:`CCTagConfig.cmake` from the cmake command line:

.. code:: shell

    cmake .. -DCCTag_DIR=$CCTAG_INSTALL/lib/cmake/CCTag/


------------



Docker image
============

A docker image can be built using the Ubuntu based :code:`Dockerfile`, which is based on nvidia/cuda image (https://hub.docker.com/r/nvidia/cuda/ )


Building the dependency image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We provide a :code:`Dockerfile_deps` containing a cuda image with all the necessary CCTag dependencies installed.

A parameter :code:`CUDA_TAG` can be passed when building the image to select the cuda version.
Similarly, :code:`OS_TAG` can be passed to select the Ubuntu version.
By default, :code:`CUDA_TAG=10.2` and :code:`OS_TAG=18.04`

For example to create the dependency image based on ubuntu 18.04 with cuda 8.0 for development, use

.. code:: shell

    docker build --build-arg CUDA_TAG=8.0 --tag alicevision/cctag-deps:cuda8.0-ubuntu18.04 -f Dockerfile_deps .

The complete list of available tags can be found on the nvidia `dockerhub page <https://hub.docker.com/r/nvidia/cuda/>`_


Building the CCTag image
~~~~~~~~~~~~~~~~~~~~~~~~

Once you built the dependency image, you can build the cctag image in the same manner using :code:`Dockerfile`:

.. code:: shell

    docker build --tag alicevision/cctag:cuda8.0-ubuntu18.04 .


Running the CCTag image
~~~~~~~~~~~~~~~~~~~~~~~

In order to run the image nvidia docker is needed: see the `installation instruction <https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)>`_.
Once installed, the docker can be run, e.g., in interactive mode with

.. code:: shell

    docker run -it --runtime=nvidia alicevision/cctag:cuda8.0-ubuntu18.04


Official images on DockeHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check the docker hub `CCTag repository <https://hub.docker.com/repository/docker/alicevision/cctag>`_ for the available images.