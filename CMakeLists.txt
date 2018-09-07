# Version 3,. is the first that supports separable compilation on Linux.
# Windows requires more work, and Mac is probably still hopeless.
# Version 3.13 is needed for boost-stacktrace
cmake_minimum_required(VERSION 3.13)

include(ExternalProject)

project(CCTag VERSION 1.0.0 LANGUAGES C CXX)

#CMakeDependentOption
option(CCTAG_SERIALIZE "Store all the output" OFF)
option(CCTAG_VISUAL_DEBUG "Enable visual debug" OFF)
option(CCTAG_NO_COUT "Disable output stream" ON)
option(CCTAG_WITH_CUDA "Compile the library with CUDA support" ON)
option(BUILD_APPS "Build the sample applications" ON)
option(CCTAG_NVCC_WARNINGS "Switch on several additional warnings for CUDA nvcc." OFF)

option(CCTAG_USE_POSITION_INDEPENDENT_CODE "Generate position independent code." ON)
option(CCTAG_ENABLE_SIMD_AVX2 "Enable AVX2 optimizations" OFF)
option(BUILD_TESTS "Build the unity tests" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT MSVC)
  set(CMAKE_BUILD_TYPE "Release")
endif()

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

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  add_definitions(/EHsc)  # Enable Exception Handling
endif()

# load local machine config <hostname>.cmake
site_name(MACHINE_NAME)

if(EXISTS "${CMAKE_SOURCE_DIR}/${MACHINE_NAME}.cmake")
  include("${MACHINE_NAME}.cmake")
else()
  message(WARNING "Compiling without ${CMAKE_SOURCE_DIR}/${MACHINE_NAME}.cmake" )
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ${CCTAG_USE_POSITION_INDEPENDENT_CODE})


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
set(BOOST_REQUIRED_COMPONENTS "atomic;chrono;date_time;filesystem;program_options;serialization;system;thread;timer")
if(WIN32)
  set(BOOST_REQUIRED_COMPONENTS "${BOOST_REQUIRED_COMPONENTS};stacktrace_windbg")
else()
  set(BOOST_REQUIRED_COMPONENTS "${BOOST_REQUIRED_COMPONENTS};stacktrace_basic")
endif()

if(BUILD_TESTS)
  set(BOOST_REQUIRED_COMPONENTS "${BOOST_REQUIRED_COMPONENTS};unit_test_framework")
  enable_testing()
  include(BoostTestHelper)
endif()

find_package(Boost 1.66.0 REQUIRED COMPONENTS ${BOOST_REQUIRED_COMPONENTS})


if(CCTAG_WITH_CUDA)
  message( STATUS "Try finding CUDA" )
  if(BUILD_SHARED_LIBS)
    message(STATUS "BUILD_SHARED_LIBS ON")
    # Need to declare CUDA_USE_STATIC_CUDA_RUNTIME as an option to ensure that it is not overwritten in FindCUDA.
    option(CUDA_USE_STATIC_CUDA_RUNTIME "Use the static version of the CUDA runtime library if available" OFF)
    set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
    # Workaround to force deactivation of cuda static runtime for cmake < 3.10
    set(CUDA_cudart_static_LIBRARY 0)
  else()
    message(STATUS "BUILD_SHARED_LIBS OFF")
    option(CUDA_USE_STATIC_CUDA_RUNTIME "Use the static version of the CUDA runtime library if available" ON)
    set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
  endif()

  find_package(CUDA 7.0 REQUIRED)

  include(CheckNvccCompilerFlag)

  set(CUDA_SEPARABLE_COMPILATION ON)

  # The following if should not be necessary, but apparently there is a bug in FindCUDA.cmake that
  # generate an empty string in the nvcc command line causing the compilation to fail.
  # see https://gitlab.kitware.com/cmake/cmake/issues/16411
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Building in debug mode")
    set(CUDA_NVCC_FLAGS_DEBUG   "${CUDA_NVCC_FLAGS_DEBUG};-G;-g")
  else()
    message(STATUS "Building in release mode")
  endif()

  if(NOT CUDA_VERSION_MAJOR LESS 9)
    set(CCTAG_CUDA_CC_LIST_BASIC 30 32 35 37 50 52 53)
  else
    set(CCTAG_CUDA_CC_LIST_BASIC 30 32 35 37 50 52 53 60 61 62 70 71 72)
  endif()
  set(CCTAG_CUDA_CC_LIST ${CCTAG_CUDA_CC_LIST_BASIC} CACHE STRING "CUDA CC versions to compile")

  # Add all requested CUDA CCs to the command line for offline compilation
  list(SORT CCTAG_CUDA_CC_LIST)
  foreach(CCTAG_CC_VERSION ${CCTAG_CUDA_CC_LIST})
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode;arch=compute_${CCTAG_CC_VERSION},code=sm_${CCTAG_CC_VERSION}")
  endforeach()

  # Use the highest request CUDA CC for CUDA JIT compilation
  list(LENGTH CCTAG_CUDA_CC_LIST CCTAG_CC_LIST_LEN)
  MATH(EXPR CCTAG_CC_LIST_LEN "${CCTAG_CC_LIST_LEN}-1")
  list(GET CCTAG_CUDA_CC_LIST ${CCTAG_CC_LIST_LEN} CCTAG_CUDA_CC_LIST_LAST)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode;arch=compute_${CCTAG_CUDA_CC_LIST_LAST},code=compute_${CCTAG_CUDA_CC_LIST_LAST}")

  if(NOT CUDA_VERSION VERSION_LESS 7.0) # cuda >= 7.0
    set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-std=c++11")
  endif()

  set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};--default-stream;per-thread")

  if(CCTAG_USE_POSITION_INDEPENDENT_CODE)
    set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-Xcompiler;-fPIC")
  endif()

  CCTAG_CHECK_NVCC_COMPILER_FLAG("--relaxed-constexpr" HAS_NVCC_RELAXED_CONSTEXPR)
  if(HAS_NVCC_RELAXED_CONSTEXPR)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--relaxed-constexpr")
  else()
    CCTAG_CHECK_NVCC_COMPILER_FLAG("--expt-relaxed-constexpr" HAS_NVCC_EXPT_RELAXED_CONSTEXPR)
    if(HAS_NVCC_EXPT_RELAXED_CONSTEXPR)
      set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--expt-relaxed-constexpr")
    endif()
  endif()

  if(NOT CUDA_VERSION VERSION_LESS 7.0) # cuda >= 7.0
    if(CCTAG_NVCC_WARNINGS)
      set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS_RELEASE};-Xptxas;-warn-lmem-usage")
      set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS_RELEASE};-Xptxas;-warn-spills")
      set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS_RELEASE};-Xptxas;--warn-on-local-memory-usage")
      set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS_RELEASE};-Xptxas;--warn-on-spills")
    endif()
  endif()

  cuda_find_library_local_first(CUDA_CUDADEVRT_LIBRARY cudadevrt "\"cudadevrt\" library")

else(CCTAG_WITH_CUDA)
  message( STATUS "Building without CUDA" )
endif(CCTAG_WITH_CUDA)

# FIND OPENCV
find_package(OpenCV REQUIRED core videoio imgproc imgcodecs)

# FIND Eigen
find_package(Eigen REQUIRED)

# FIND Intel TBB
# With MVSC, CMAKE_BUILD_TYPE will always be None, so TBB_USE_DEBUG_BUILD will always be false.
string(COMPARE EQUAL "${CMAKE_BUILD_TYPE}" Debug TBB_USE_DEBUG_BUILD)
find_package(TBB 4 REQUIRED)

add_subdirectory(src)

########### Add uninstall target ###############
CONFIGURE_FILE(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake"
  IMMEDIATE @ONLY)
ADD_CUSTOM_TARGET(uninstall
  "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake")
