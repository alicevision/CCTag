# IF(COMMAND cmake_policy)
cmake_policy(SET CMP0054 OLD) # behaviour keywords in CMake if tests
# ENDIF(COMMAND cmake_policy)

set( CCTag_cpp
        ./cctag/Bresenham.cpp
        ./cctag/CCTag.cpp
        ./cctag/CCTagFlowComponent.cpp
        ./cctag/CCTagMarkersBank.cpp
        ./cctag/Candidate.cpp
        ./cctag/Canny.cpp
        ./cctag/DataSerialization.cpp
        ./cctag/Detection.cpp
        ./cctag/EdgePoint.cpp
        ./cctag/EllipseGrowing.cpp
        ./cctag/Fitting.cpp
        ./cctag/ICCTag.cpp
        ./cctag/Identification.cpp
        ./cctag/ImagePyramid.cpp
        ./cctag/Level.cpp
        ./cctag/Multiresolution.cpp
        ./cctag/Params.cpp
        ./cctag/Statistic.cpp
        ./cctag/SubPixEdgeOptimizer.cpp
        ./cctag/Types.cpp
        ./cctag/Vote.cpp
        ./cctag/algebra/matrix/Operation.cpp
        ./cctag/filter/cvRecode.cpp
        ./cctag/filter/thinning.cpp
        ./cctag/geometry/2DTransform.cpp
        ./cctag/geometry/Circle.cpp
        ./cctag/geometry/Distance.cpp
        ./cctag/geometry/Ellipse.cpp
        ./cctag/geometry/EllipseFromPoints.cpp
        ./cctag/utils/FileDebug.cpp
        ./cctag/utils/LogTime.cpp
        ./cctag/utils/Talk.cpp
        ./cctag/utils/VisualDebug.cpp)

if(CCTAG_WITH_CUDA)

  set(CUDA_cpp
        ./cctag/cuda/debug_macros.cpp
        ./cctag/cuda/device_prop.cpp)

  set(CUDA_cu
        ./cctag/cuda/assist.cu
        ./cctag/cuda/cmp_list.cu
        ./cctag/cuda/debug_image.cu
        ./cctag/cuda/debug_is_on_edge.cu
        ./cctag/cuda/frame.cu
        ./cctag/cuda/frame_01_tex.cu
        ./cctag/cuda/frame_02_gaussian.cu
        ./cctag/cuda/frame_03_magmap.cu
        ./cctag/cuda/frame_04_hyst.cu
        ./cctag/cuda/frame_05_thin.cu
        ./cctag/cuda/frame_06_graddesc.cu
        ./cctag/cuda/frame_07_vote.cu
        ./cctag/cuda/frame_07a_vote_line.cu
        ./cctag/cuda/frame_07b_vote_sort_uniq_thrust.cu
        ./cctag/cuda/frame_07c_eval.cu
        ./cctag/cuda/frame_07d_vote_if.cu
        ./cctag/cuda/frame_07e_download.cu
        ./cctag/cuda/frame_alloc.cu
        ./cctag/cuda/frame_debug.cu
        ./cctag/cuda/frame_export.cu
        ./cctag/cuda/framemeta.cu
        ./cctag/cuda/frameparam.cu
        ./cctag/cuda/geom_ellipse.cu
        ./cctag/cuda/geom_matrix.cu
        ./cctag/cuda/keep_time.cu
        ./cctag/cuda/pinned_counters.cu
        ./cctag/cuda/ptrstep.cu
        ./cctag/cuda/recursive_sweep.cu
        ./cctag/cuda/tag.cu
        ./cctag/cuda/tag_identify.cu
        ./cctag/cuda/tag_threads.cu
        ./cctag/cuda/triple_point.cu)

  # add_definitions("-DSM_TARGETS=-gencode=arch=compute_52,code=\"sm_52,compute_52\"")

  # set(CUDA_SEPARABLE_COMPILATION ON)
  # set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
  # set(BUILD_SHARED_LIBS ON)

  set(CUDA_NVCC_FLAGS  "${CUDA_NVCC_FLAGS};-DCUB_CDP")

  # this must go before CUDA_ADD_LIBRARY otherwise we won't be able to add it
  # after
  # https://github.com/Kitware/CMake/blob/master/Modules/FindCUDA.cmake#L147
  # INCLUDE_DIRECTORIES -- "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>"
  include_directories(${CMAKE_CURRENT_LIST_DIR}
                      ${opencv_INCLUDE_DIRS} ${Eigen3_INCLUDE_DIR}
		      ${TBB_INCLUDE_DIRS}
                      "${CMAKE_SOURCE_DIR}/src")

  CUDA_ADD_LIBRARY(CCTag ${CUDA_cpp} ${CUDA_cu} ${CCTag_cpp})

  # since we used CUDA_ADD_LIBRARY we cannot use PUBLIC or PRIVATE here
  target_link_libraries(CCTag
                       ${opencv_LIBS}
                       boost_date_time boost_chrono boost_thread boost_serialization boost_system boost_filesystem boost_atomic boost_program_options boost_timer
                       ${Eigen3_LIBRARIES}
                       ${TBB_LIBRARIES} ${CUDA_CUDADEVRT_LIBRARY})

  if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
   target_link_libraries(CCTag pthread dl)
  endif()

  target_compile_definitions(CCTag
                         PUBLIC -DCCTAG_WITH_CUDA -DCUB_CDP
                         PRIVATE ${TBB_DEFINITIONS})

  if(CCTAG_HAVE_SHFL_DOWN_SYNC)
    target_compile_definitions(CCTag PRIVATE "-DCCTAG_HAVE_SHFL_DOWN_SYNC")
  endif(CCTAG_HAVE_SHFL_DOWN_SYNC)

  if(CCTAG_NO_THRUST_COPY_IF)
    target_compile_definitions(CCTag PRIVATE "-DCCTAG_NO_THRUST_COPY_IF")
  endif()

  # This is nececessary for the CCTagConfig.cmake to correctly export the
  # includes, always because we used CUDA_ADD_LIBRARY
  set_target_properties(CCTag PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
       "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>;${CUDA_INCLUDE_DIRS};${Boost_INCLUDE_DIRS};${Eigen3_INCLUDE_DIR};${opencv_INCLUDE_DIRS}")
  get_target_property(testprop CCTag INTERFACE_INCLUDE_DIRECTORIES )
  message(STATUS "testprop: ${testprop}")

else() # without CUDA

  set(CUDA_cpp "")

  message( STATUS "Declare CCTag library" )
  add_library(CCTag ${CCTag_cpp})
  set_target_properties(CCTag PROPERTIES VERSION ${PROJECT_VERSION})

  target_include_directories(CCTag
                           PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>"
                                  "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>"
                           PUBLIC ${Boost_INCLUDE_DIRS} ${Eigen3_INCLUDE_DIR} ${opencv_INCLUDE_DIRS}
                           PRIVATE ${TBB_INCLUDE_DIRS})

  get_target_property(testprop CCTag INTERFACE_INCLUDE_DIRECTORIES )
  message(STATUS "testprop: ${testprop}")

  target_compile_definitions(CCTag
                             PRIVATE ${TBB_DEFINITIONS})

  target_link_libraries(CCTag
                        PUBLIC
                          ${opencv_LIBS}
                          ${Eigen3_LIBRARIES}
                          boost_atomic boost_chrono boost_date_time boost_filesystem boost_serialization boost_system boost_thread boost_timer
                        PRIVATE
                          ${TBB_LIBRARIES})

  if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    target_link_libraries(CCTag PRIVATE pthread dl)
  endif()

endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  target_compile_definitions(CCTag PUBLIC "-DNOMINMAX")
endif()

# enable serialization
if(CCTAG_SERIALIZE)
  target_compile_definitions(CCTag PUBLIC "-DCCTAG_SERIALIZE")
endif(CCTAG_SERIALIZE)
# Disable output stream
if(CCTAG_NO_COUT)
  target_compile_definitions(CCTag PUBLIC "-DCCTAG_NO_COUT")
endif(CCTAG_NO_COUT)
# Enable visual debug
if(CCTAG_VISUAL_DEBUG)
  target_compile_definitions(CCTag PRIVATE "-DCCTAG_VISUAL_DEBUG")
endif(CCTAG_VISUAL_DEBUG)


# EXPORTING THE LIBRARY
#
# place to put the cmake-related files
set(config_install_dir "lib/cmake/${PROJECT_NAME}")
# include directory for install
set(include_install_dir "include")
# the name for the generated config.hpp
set(confighpp_name "cctag_config.hpp")
# the name for the generated header version file
set(versionhpp_name "version.hpp")
# where to install the generated files
set(install_dir_generated "${include_install_dir}/cctag")

# build directory containing the generated files
set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")

# Configuration
set(version_config "${generated_dir}/${PROJECT_NAME}ConfigVersion.cmake")
set(project_config "${generated_dir}/${PROJECT_NAME}Config.cmake")
set(targets_export_name "${PROJECT_NAME}Targets")
set(namespace "${PROJECT_NAME}::")

# Include module with fuction 'write_basic_package_version_file'
include(CMakePackageConfigHelpers)

# Configure '<PROJECT-NAME>ConfigVersion.cmake'
# Note: major version number must be the same as requested
write_basic_package_version_file("${version_config}" COMPATIBILITY SameMajorVersion)

# Configure '<PROJECT-NAME>Config.cmake'
# Use variables:
#   * targets_export_name
#   * PROJECT_NAME
configure_package_config_file("${PROJECT_SOURCE_DIR}/cmake/Config.cmake.in"
                              "${project_config}"
                              INSTALL_DESTINATION "${config_install_dir}")

# config file
configure_file("${PROJECT_SOURCE_DIR}/cmake/config.hpp.in" ${generated_dir}/${confighpp_name} @ONLY)

# version file
configure_file("${PROJECT_SOURCE_DIR}/cmake/version.hpp.in" ${generated_dir}/${versionhpp_name} @ONLY)

# Targets:
#   * <prefix>/lib/libCCTag.a
#   * header location after install: <prefix>/include/
#   * headers can be included by C++ code `#include <popsift/popsift.h>`
install(TARGETS CCTag
        EXPORT "${targets_export_name}"
        LIBRARY DESTINATION "lib"
        ARCHIVE DESTINATION "lib"
        RUNTIME DESTINATION "bin"
        INCLUDES DESTINATION "${include_install_dir}")

# Headers:


if(CCTAG_WITH_CUDA)
  install(DIRECTORY "cctag"
          DESTINATION "${include_install_dir}"
          FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")
else()
  install(DIRECTORY "cctag"
          DESTINATION "${include_install_dir}"
          FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
          PATTERN "cuda/*" EXCLUDE)
endif()

# install the config and version generated files
install( FILES "${generated_dir}/${confighpp_name}"
    DESTINATION "${install_dir_generated}")

install( FILES "${generated_dir}/${versionhpp_name}"
    DESTINATION "${install_dir_generated}")

# install cmake config files
# Config
#   * <prefix>/lib/cmake/${PROJECT_NAME}/${PROJECT_NAME}Config.cmake
#   * <prefix>/lib/cmake/${PROJECT_NAME}${PROJECT_NAME}ConfigVersion.cmake
install(FILES "${project_config}" "${version_config}"
        DESTINATION "${config_install_dir}")

# Config
#   * <prefix>/lib/cmake/${PROJECT_NAME}/${PROJECT_NAME}Targets.cmake
install(EXPORT "${targets_export_name}"
        NAMESPACE "${namespace}"
        DESTINATION "${config_install_dir}")

if(BUILD_APPS)
  add_subdirectory(applications)
endif()

if(BUILD_TESTS)
    add_subdirectory(cctag/test)
    add_subdirectory(cctag/geometry/test)
endif()
