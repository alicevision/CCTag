# CCTag Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Enable -faligned-new when CCTAG_EIGEN_NO_ALIGN is not set on GCC >= 7.1 [PR](https://github.com/alicevision/CCTag/pull/193)

### Changed

### Fixed

### Removed

## [1.0.2] - 2022-07-14

### Added
- Added documentation for conan [PR](https://github.com/alicevision/CCTag/pull/183)

### Fixed
- Fixed potential out of bound in debug mode [PR](https://github.com/alicevision/CCTag/pull/180)

## [1.0.1] - 2021-10-08

### Changed

- Renamed the marker files to be 0-based [PR](https://github.com/alicevision/CCTag/pull/165) 

### Fixed

- Fixed missing math module for boost [PR](https://github.com/alicevision/CCTag/pull/168)
- Fixed compilation errors for newer version of boost on windows [PR](https://github.com/alicevision/CCTag/pull/166)
- Removed old legacy defines for cuda and boost no more needed since the switch to c++14 [PR](https://github.com/alicevision/CCTag/pull/174)


## [1.0.0] - 2021-06-24

 - Support for OpenCV 3.4.9 [PR](https://github.com/alicevision/CCTag/pull/121)
 - Switch to C++14 standard [PR](https://github.com/alicevision/CCTag/pull/155)


## 2019

 - Remove Cuda Cub dependency [PR](https://github.com/alicevision/CCTag/pull/110)
 - Bug fix: out of range image access during identification [PR](https://github.com/alicevision/CCTag/pull/117)
 - Bug fix on big markers (multi-resolution detection) [PR](https://github.com/alicevision/CCTag/pull/116)
 - Bug fix: avoid access to empty vector during vote [PR](https://github.com/alicevision/CCTag/pull/115)
 - CMake: export all symbols when building shared libs on Windows [PR](https://github.com/alicevision/CCTag/pull/112)
 - CMake: Improve management of CUDA Compute Capabilities flags [PR](https://github.com/alicevision/CCTag/pull/109)
 - Compatibility with Opencv 4 [PR](https://github.com/alicevision/CCTag/pull/104)


## 2018

 - First Windows version (CPU only) [PR](https://github.com/alicevision/CCTag/pull/78)
 - Docker: add support for docker [PR](https://github.com/alicevision/CCTag/pull/84)
 - Improvements for ellipse fitting
[PR](https://github.com/alicevision/CCTag/pull/66)
 - Modernizing code to C++11
[PR](https://github.com/alicevision/CCTag/pull/64)


## 2017

 - Use Thrust for cuda >= 8
[PR](https://github.com/alicevision/CCTag/pull/62)
 - Minor code cleaning
[PR](https://github.com/alicevision/CCTag/pull/61)
 - Add limit to edgepoint d-to-h transfer
[PR](https://github.com/alicevision/CCTag/pull/53)
 - Bug fix in NearbyPoint
[PR](https://github.com/alicevision/CCTag/pull/46)


## 2016

 - Switch to modern version of CMake [PR](https://github.com/alicevision/CCTag/pull/40)
 - CVPR Publication https://hal.archives-ouvertes.fr/hal-01420665/document
 - Application: Show detected markers in video and live mode
[PR](https://github.com/alicevision/CCTag/pull/33)
 - Allow to extract CCTags from multiple images in parallel with multiple cuda streams
[PR](https://github.com/alicevision/CCTag/pull/32)
[PR](https://github.com/alicevision/CCTag/pull/31)
 - Continuous integration on Travis [PR](https://github.com/alicevision/CCTag/pull/27)
 - Remove Blas/Lapack/Optpp dependencies
 - Remove Ceres dependency
 - GPU implementation


## 2015

 - First open-source release
 - CPU Optimizations
 - Expose critical parameters
 - CMake build system


## 2014

 - Thesis Defence by Lilian Calvet under the direction of Vincent Charvillat and Pierre Gurdjos
   "Three-dimensional reconstruction methods integrating cyclic points: application to camera tracking" (http://www.theses.fr/2014INPT0002)
 - CCTag detection on CPU

