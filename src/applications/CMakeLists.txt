cmake_minimum_required(VERSION 3.13)

include(ExternalProject)

project(CCTagApplications LANGUAGES C CXX)

set(CMAKE_BUILD_TYPE Release)

#Find needed dependencies
# CCTAG
if(TARGET CCTag)
  add_library(CCTag::CCTag ALIAS CCTag)
  message(STATUS "CCTAG already there")
else()
  # Add NO_CMAKE_BUILDS_PATH for windows if using CMake-GUI to build packages
  # to avoid searching in temporary build directory of Foo project
  # See 5:
  #    * http://www.cmake.org/cmake/help/v3.0/command/find_package.html
  find_package(CCTag CONFIG REQUIRED)
endif()

# BOOST
if(NOT TARGET Boost::Boost)
  set(BOOST_REQUIRED_COMPONENTS "date_time;chrono;thread;serialization;system;filesystem;atomic;program_options;timer")
  if(WIN32)
    set(BOOST_REQUIRED_COMPONENTS "${BOOST_REQUIRED_COMPONENTS};stacktrace_windbg")
  else()
    set(BOOST_REQUIRED_COMPONENTS "${BOOST_REQUIRED_COMPONENTS};stacktrace_basic")
  endif()

  find_package(Boost 1.66.0 REQUIRED COMPONENTS ${BOOST_REQUIRED_COMPONENTS} QUIET)
else()
  message(STATUS "BOOST already there")
endif()

# OPENCV
find_package(opencv REQUIRED core videoio imgproc imgcodecs highgui)

# TBB
if(NOT TBB_FOUND)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")
  find_package(TBB 4 REQUIRED)
else()
message(STATUS "TBB already there")
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(CCTagRegression_cpp
  ./regression/main.cpp
  ./regression/TestLog.cpp
  ./regression/Regression.cpp)

set(CCTagSimulation_cpp
  ./simulation/main.cpp)

get_target_property(testprop CCTag::CCTag INTERFACE_INCLUDE_DIRECTORIES)

set(CCTagDetect_cpp ./detection/main.cpp ./detection/CmdLine.cpp)
add_executable(detection ${CCTagDetect_cpp})

find_package(DevIL COMPONENTS IL ILU) # yields IL_FOUND, IL_LIBRARIES, IL_INCLUDE_DIR

target_include_directories(detection PUBLIC
  ${opencv_INCLUDE_DIRS}
  ${TBB_INCLUDE_DIRS}
  )
target_link_libraries(detection PUBLIC
  CCTag::CCTag
  ${TBB_tbb_LIBRARY_RELEASE}
  ${opencv_LIBS}
  boost_filesystem boost_program_options boost_timer
  )

if(IL_FOUND OR DevIL_FOUND)
  message(STATUS "DevIL found")
  target_compile_options(detection PRIVATE -DUSE_DEVIL)
  target_include_directories(detection PUBLIC
    ${IL_INCLUDE_DIR}
    )
  target_link_libraries(detection PUBLIC
    ${IL_LIBRARIES}
    ${ILU_LIBRARIES}
    )
else()
  message(STATUS "DevIL not found")
endif()

add_executable(regression ${CCTagRegression_cpp})
target_include_directories(regression PUBLIC ${opencv_INCLUDE_DIRS})
target_link_libraries(regression PUBLIC
  CCTag::CCTag
  ${opencv_LIBS}
  boost_program_options boost_serialization
  )

add_executable(simulation ${CCTagSimulation_cpp})
target_include_directories(simulation PUBLIC ${opencv_INCLUDE_DIRS})
target_link_libraries(simulation PUBLIC ${opencv_LIBS})

install(TARGETS detection regression simulation DESTINATION bin)
