# Version 3,4 is the first that supports separable compilation on Linux.
# Windows requires more work, and Mac is probably still hopeless.
cmake_minimum_required(VERSION 3.4)

include(ExternalProject)

project(CCTag VERSION 1.0.0 LANGUAGES C CXX)

#CMakeDependentOption
option(CCTAG_SERIALIZE "Store all the output" OFF)
option(VISUAL_DEBUG "Enable visual debug" OFF)
option(CCTAG_NO_COUT "Disable output stream" ON)
option(WITH_CUDA "Compile the library with CUDA support" ON)
option(BUILD_APPS "Build the sample applications" ON)

option(CCTAG_USE_POSITION_INDEPENDENT_CODE "Generate position independent code." ON)
option(CCTAG_ENABLE_SIMD_AVX2 "Enable AVX2 optimizations" OFF)

if(CCTAG_ENABLE_SIMD_AVX2)
  if(CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
    add_definitions(-mavx2)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    add_definitions(/QxAVX2)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_definitions(/arch:AVX2)
  endif()
  message(STATUS "CCTAG: AVX2 optimizations enabled.")
endif()

# Cuda architecture values
set(CCTAG_SM_ARCH 100)
set(CCTAG_SM 100)

# load local machine config <hostname>.cmake
site_name(MACHINE_NAME)

if(EXISTS "${CMAKE_SOURCE_DIR}/${MACHINE_NAME}.cmake")
  include("${MACHINE_NAME}.cmake")
else()
  message(WARNING "Compiling without ${CMAKE_SOURCE_DIR}/${MACHINE_NAME}.cmake" )
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall ")

# set the path where we can find the findXXX.cmake
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # -Wno-c++11-narrowing to solve "non-type template argument evaluates to -1" error in boost::gil
  # -Wno-deprecated-register to avoid the noise of some older eigen versions
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++11-extensions -Wno-c++11-narrowing -Wno-deprecated-register")
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

  # The following if should not be necessary, but apparently there is a bug in FindCUDA.cmake that
  # generate an empty string in the nvcc command line causing the compilation to fail.
  # see https://gitlab.kitware.com/cmake/cmake/issues/16411
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Building in debug mode")
    set(CUDA_NVCC_FLAGS_DEBUG   "${CUDA_NVCC_FLAGS_DEBUG};-G;-g")
  else()
    message(STATUS "Building in release mode")
    # set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE};-O3;-DNDEBUG")
    # set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE};-G;-g;-DNDEBUG")
  endif()

  #set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-gencode;arch=compute_20,code=sm_20")
  #set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-gencode;arch=compute_30,code=sm_30")
  set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35")
  set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-gencode;arch=compute_50,code=sm_50")
  set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-gencode;arch=compute_52,code=sm_52")
  set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-gencode=arch=compute_52,code=compute_52")

  if(NOT CUDA_VERSION VERSION_LESS 8.0) # cuda >= 8.0
    set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-gencode=arch=compute_60,code=sm_60")
    set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-gencode=arch=compute_61,code=sm_61")
    set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-gencode=arch=compute_62,code=sm_62")
  endif()

  if(NOT CUDA_VERSION VERSION_LESS 7.0) # cuda >= 7.0
    set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-std=c++11")
  endif()

  set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};--default-stream;per-thread")

  if(CCTAG_USE_POSITION_INDEPENDENT_CODE)
    set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-Xcompiler;-fPIC")
  endif()

  # MORE NOISE:
  # set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-gencode=arch=compute_52,code=sm_52;-gencode=arch=compute_52,code=compute_52;--default-stream;per-thread;-Xptxas;-v")
  # DOES NOT WORK UNTIL CUDA 7.5:
  # set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-gencode=arch=compute_52,code=sm_52;-gencode=arch=compute_52,code=compute_52;--default-stream;per-thread;-Xptxas;--warn-on-local-memory-usage;-Xptxas;--warn-on-spills")
  cuda_find_library_local_first(CUDA_CUDADEVRT_LIBRARY cudadevrt "\"cudadevrt\" library")

else(WITH_CUDA)
  message( STATUS "Building without CUDA" )
endif(WITH_CUDA)

# FIND OPENCV
find_package(OpenCV REQUIRED core videoio imgproc imgcodecs)

# FIND Eigen
find_package(Eigen REQUIRED)

# FIND Intel TBB
find_package(TBB 4 REQUIRED)

add_subdirectory(src)

########### Add uninstall target ###############
CONFIGURE_FILE(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake"
  IMMEDIATE @ONLY)
ADD_CUSTOM_TARGET(uninstall
  "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake")
