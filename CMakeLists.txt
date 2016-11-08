# Version 3,4 is the first that supports separable compilation on Linux.
# Windows requires more work, and Mac is probably still hopeless.
cmake_minimum_required(VERSION 3.4)

include(ExternalProject)

project(CCTag VERSION 1.0.0 LANGUAGES C CXX)

#CMakeDependentOption
option(CCTAG_SERIALIZE "Store all the output" OFF)
option(CCTAG_NO_COUT "Disable output stream" ON)
option(WITH_CUDA "Compile the library with CUDA support" ON)
option(COMPILE_STATIC_CCTAG_LIBRARY "Generate static library" ON)

# Cuda architecture valeues
set(CCTAG_SM_ARCH 100)
set(CCTAG_SM 100)

# load local machine config <hostname>.cmake
site_name(MACHINE_NAME)

if(EXISTS "${CMAKE_SOURCE_DIR}/${MACHINE_NAME}.cmake")
  include("${MACHINE_NAME}.cmake")
else()
  message(WARNING "Compiling without ${CMAKE_SOURCE_DIR}/${MACHINE_NAME}.cmake" )
endif()

# I`m really sorry, but this makes CUDA compilation impossible
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
  
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall ")

# set the path where we can find the findXXX.cmake
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # -Wno-c++11-narrowing to solve "non-type template argument evaluates to -1" error in boost::gil
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++11-extensions -Wno-c++11-narrowing")
endif()

if(APPLE)

  # avoid the cmake policy warning about @rpath in MacOSX
  cmake_policy(SET CMP0042 NEW)

  SET(CMAKE_MACOSX_RPATH TRUE) # initialize the MACOSX_RPATH property on all targets
  SET(CMAKE_SKIP_BUILD_RPATH  FALSE) # don't skip the full RPATH for the build tree
  # SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) # when building, don't use the install RPATH already
  SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE) # when building, use the install RPATH already
                                           # probably not needed
  # SET(CMAKE_INSTALL_RPATH "") # the RPATH to be used when installing
  SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE) # LC_RPATH for CUDA and OpenCV etc written into executable
endif(APPLE)

# FIND BOOST
# TODO: determine minimal version and minimal components
find_package(Boost 1.53.0 REQUIRED
    COMPONENTS date_time chrono thread serialization system filesystem atomic program_options timer)

if(WITH_CUDA)
  message( STATUS "Try finding CUDA" )
  find_package(CUDA 7.0 REQUIRED)
  set(CUDA_SEPARABLE_COMPILATION ON)
  #    include_directories(${CUDA_INCLUDE_DIRS})
  #    add_definitions("-DWITH_CUDA")
  set(CUDA_NVCC_FLAGS_DEBUG   "${CUDA_NVCC_FLAGS_DEBUG};-G;-g")
  set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE};-O3;-DNDEBUG")
  # set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE};-G;-g;-DNDEBUG")
  set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};--std=c++11")
  set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-gencode=arch=compute_52,code=sm_52")
  set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-gencode=arch=compute_52,code=compute_52")
  set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};--default-stream;per-thread")
  # MORE NOISE:
  # set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-gencode=arch=compute_52,code=sm_52;-gencode=arch=compute_52,code=compute_52;--default-stream;per-thread;-Xptxas;-v")
  # DOES NOT WORK UNTIL CUDA 7.5:
  # set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-gencode=arch=compute_52,code=sm_52;-gencode=arch=compute_52,code=compute_52;--default-stream;per-thread;-Xptxas;--warn-on-local-memory-usage;-Xptxas;--warn-on-spills")
  cuda_find_library_local_first(CUDA_CUDADEVRT_LIBRARY cudadevrt "\"cudadevrt\" library")

else(WITH_CUDA)
  message( STATUS "Building without CUDA" )
endif(WITH_CUDA)

# FIND OPENCV
find_package(OpenCV REQUIRED)

# FIND Eigen
find_package(Eigen REQUIRED)

# FIND PNG
find_package(PNG REQUIRED)

# FIND JPEG
find_package(JPEG REQUIRED)

# FIND Intel TBB
find_package(TBB 4 REQUIRED)

add_subdirectory(src)

