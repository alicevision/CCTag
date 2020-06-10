# Version 3,4 is the first that supports separable compilation on Linux.
# Windows requires more work, and Mac is probably still hopeless.
# Version 3.13 is needed for boost-stacktrace
cmake_minimum_required(VERSION 3.13)

include(ExternalProject)

project(CCTag VERSION 1.0.0 LANGUAGES C CXX)

# Set build path as a folder named as the platform (linux, windows, darwin...) plus the processor type
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")

include(GNUInstallDirs)

#CMakeDependentOption
option(CCTAG_SERIALIZE "Store all the output" OFF)
option(CCTAG_VISUAL_DEBUG "Enable visual debug" OFF)
option(CCTAG_NO_COUT "Disable output stream" ON)
option(CCTAG_WITH_CUDA "Compile the library with CUDA support" ON)
option(CCTAG_BUILD_APPS "Build the sample applications" ON)
option(CCTAG_CUDA_CC_CURRENT_ONLY "Set to on to build only for the current machine's CC" OFF)
option(CCTAG_NVCC_WARNINGS "Switch on several additional warnings for CUDA nvcc." OFF)
option(CCTAG_EIGEN_NO_ALIGN "Disable Eigen alignment" ON)

option(CCTAG_USE_POSITION_INDEPENDENT_CODE "Generate position independent code." ON)
option(CCTAG_ENABLE_SIMD_AVX2 "Enable AVX2 optimizations" OFF)
option(CCTAG_BUILD_TESTS "Build the unity tests" ON)
option(CCTAG_BUILD_DOC "Build documentation" OFF)
option(CCTAG_NO_THRUST_COPY_IF "Do not use thrust::copy_if() on GPU. There may be a bug on CUDA 7 with GTX 980, 980Ti and 1080" OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

# set(CMAKE_BUILD_TYPE Debug)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
  message(STATUS "Build type not set, building in Release configuration")
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
else()
  message(STATUS "Building in ${CMAKE_BUILD_TYPE} configuration")
endif()

# ensure the proper linker flags when building the static version on MSVC
if(MSVC AND NOT BUILD_SHARED_LIBS)
  foreach(config "DEBUG" "RELEASE" "MINSIZEREL" "RELWITHDEBINFO")
    string(REPLACE /MD /MT CMAKE_C_FLAGS_${config} "${CMAKE_C_FLAGS_${config}}")
    string(REPLACE /MD /MT CMAKE_CXX_FLAGS_${config} "${CMAKE_CXX_FLAGS_${config}}")
    message(STATUS "CMAKE_C_FLAGS_${config} ${CMAKE_C_FLAGS_${config}}")
    message(STATUS "CMAKE_CXX_FLAGS_${config} ${CMAKE_CXX_FLAGS_${config}}")
  endforeach()
endif()

# this is to ensure that on MSVC the flags for the linker are properly propagate even to the intermediate
# linking step. This seems not the case e.g. on vcpkg using ninja build.
if(MSVC AND CMAKE_GENERATOR MATCHES "Ninja")
  if(BUILD_SHARED_LIBS)
    set(CCTAG_MVSC_LINKER "/MD")
  else()
    set(CCTAG_MVSC_LINKER "/MT")
  endif()
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      set(CCTAG_MVSC_LINKER "${CCTAG_MVSC_LINKER}d")
  endif()
  list(APPEND CUDA_NVCC_FLAGS   -Xcompiler ${CCTAG_MVSC_LINKER})
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


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

if(MSVC)
  add_definitions(/EHsc)  # Enable Exception Handling
endif()

if(NOT MSVC)
  set(CMAKE_POSITION_INDEPENDENT_CODE ${CCTAG_USE_POSITION_INDEPENDENT_CODE})
endif()


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

if(CCTAG_BUILD_TESTS)
  set(BOOST_REQUIRED_COMPONENTS "${BOOST_REQUIRED_COMPONENTS};unit_test_framework")
  enable_testing()
  include(BoostTestHelper)
endif()

find_package(Boost 1.66.0 REQUIRED COMPONENTS ${BOOST_REQUIRED_COMPONENTS})

if(BUILD_SHARED_LIBS)
  if(WIN32)
    # Export all symbols from the dynamic libraries by default (avoid dllexport markup)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
  endif()
endif()

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

  set(CCTAG_CUDA_CC_LIST_INIT0 3.5 3.7 5.0 5.2)
  if( CUDA_VERSION VERSION_GREATER_EQUAL "9.0" )
    list(APPEND CCTAG_CUDA_CC_LIST_INIT0 6.0 6.1 7.0)
  endif()
  if( CUDA_VERSION VERSION_GREATER_EQUAL "10.0" )
    list(APPEND CCTAG_CUDA_CC_LIST_INIT0 7.5)
  endif()
  set(CCTAG_CUDA_CC_LIST_INIT ${CCTAG_CUDA_CC_LIST_INIT0} CACHE STRING "CUDA CCs as compile targets")

  if(CCTAG_CUDA_CC_CURRENT_ONLY)
    set(CCTAG_CUDA_CC_LIST_BASIC Auto)
  else()
    set(CCTAG_CUDA_CC_LIST_BASIC ${CCTAG_CUDA_CC_LIST_INIT})
  endif()
  CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS ${CCTAG_CUDA_CC_LIST_BASIC})
  LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})

  if(NOT MSVC)
    set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-std=c++11")
  endif()
  set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};--default-stream;per-thread")

  if(CCTAG_USE_POSITION_INDEPENDENT_CODE AND NOT MSVC)
    set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS};-Xcompiler;-fPIC")
  endif()

  CCTAG_CHECK_NVCC_COMPILER_FLAG("--expt-relaxed-constexpr" HAS_NVCC_EXPT_RELAXED_CONSTEXPR)
  if(HAS_NVCC_EXPT_RELAXED_CONSTEXPR)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--expt-relaxed-constexpr")
  else()
    CCTAG_CHECK_NVCC_COMPILER_FLAG("--relaxed-constexpr" HAS_NVCC_RELAXED_CONSTEXPR)
    if(HAS_NVCC_RELAXED_CONSTEXPR)
      set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--relaxed-constexpr")
    endif()
  endif()

  if(CCTAG_NVCC_WARNINGS)
    list(APPEND CUDA_NVCC_FLAGS -Xptxas --warn-on-local-memory-usage)
    list(APPEND CUDA_NVCC_FLAGS -Xptxas --warn-on-spills)
  endif()

  cuda_find_library_local_first(CUDA_CUDADEVRT_LIBRARY cudadevrt "\"cudadevrt\" library")

  if( ( CUDA_VERSION VERSION_EQUAL "9.0" ) OR ( CUDA_VERSION VERSION_GREATER "9.0") )
    set(CCTAG_HAVE_SHFL_DOWN_SYNC   1)
  else()
    set(CCTAG_HAVE_SHFL_DOWN_SYNC   0)
  endif()
else(CCTAG_WITH_CUDA)
  message( STATUS "Building without CUDA" )
endif(CCTAG_WITH_CUDA)

# FIND OPENCV
find_package(OpenCV REQUIRED core videoio imgproc imgcodecs)

# FIND Eigen
set(CCTAG_EIGEN_REQUIRED_VERSION 3.3.4)
if(MSVC AND CCTAG_WITH_CUDA)
  set(CCTAG_EIGEN_REQUIRED_VERSION 3.3.9)
  message(WARNING "Building CCTag with Cuda support under windows requires Eigen >= ${CCTAG_EIGEN_REQUIRED_VERSION}")
endif()
find_package(Eigen3 ${CCTAG_EIGEN_REQUIRED_VERSION} REQUIRED)
message(STATUS "Found Eigen: version ${Eigen3_VERSION}")
if(CCTAG_EIGEN_NO_ALIGN)
  set(AV_EIGEN_DEFINITIONS -DEIGEN_MAX_ALIGN_BYTES=0 -DEIGEN_MAX_STATIC_ALIGN_BYTES=0)
endif()

# FIND Intel TBB
# With MVSC, CMAKE_BUILD_TYPE will always be None, so TBB_USE_DEBUG_BUILD will always be false.
string(COMPARE EQUAL "${CMAKE_BUILD_TYPE}" Debug TBB_USE_DEBUG_BUILD)
find_package(TBB REQUIRED)

add_subdirectory(src)

if(CCTAG_BUILD_DOC)
 add_subdirectory(doc)
endif()

########### Add uninstall target ###############
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake"
  IMMEDIATE @ONLY)
add_custom_target(cctag_uninstall
  "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake")


######################################
# SUMMARY
######################################
message("\n")
message("******************************************")
message("Building configuration:\n")
message(STATUS "CCTag version: " ${PROJECT_VERSION})
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})
message(STATUS "Build Shared libs: " ${BUILD_SHARED_LIBS})
message(STATUS "Build applications: " ${CCTAG_BUILD_APPS})
message(STATUS "Build tests: " ${CCTAG_BUILD_TESTS})
message(STATUS "Build documentation: " ${CCTAG_BUILD_DOC})
message(STATUS "Cuda support: " ${CCTAG_WITH_CUDA})
if(CCTAG_WITH_CUDA)
  message(STATUS "Compiling for CUDA CCs: ${ARCH_FLAGS}")
endif()
message(STATUS "Disable Eigen alignment: " ${CCTAG_EIGEN_NO_ALIGN})
message(STATUS "Enable AVX2 optimizations: " ${CCTAG_ENABLE_SIMD_AVX2})
message(STATUS "[debug] Serialize all the output: " ${CCTAG_SERIALIZE})
message(STATUS "[debug] Enable visual debug: " ${CCTAG_VISUAL_DEBUG})
message(STATUS "[debug] Disable output stream: " ${CCTAG_NO_COUT})
message(STATUS "[debug] nvcc additional warnings: " ${CCTAG_NVCC_WARNINGS})
message(STATUS "Install path: " ${CMAKE_INSTALL_PREFIX})
message("\n******************************************")
message("\n")
