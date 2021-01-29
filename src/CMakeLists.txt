
# build directory containing the generated files
set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")

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
        ./cctag/utils/Accum.cpp
        ./cctag/utils/FileDebug.cpp
        ./cctag/utils/LogTime.cpp
        ./cctag/utils/Talk.cpp
        ./cctag/utils/VisualDebug.cpp)

set( CCTag_HEADERS
        ./cctag/Bresenham.hpp
        ./cctag/CCTag.hpp
        ./cctag/CCTagFlowComponent.hpp
        ./cctag/CCTagMarkersBank.hpp
        ./cctag/Candidate.hpp
        ./cctag/Canny.hpp
        ./cctag/Colors.hpp
        ./cctag/DataSerialization.hpp
        ./cctag/Detection.hpp
        ./cctag/EdgePoint.hpp
        ./cctag/EllipseGrowing.hpp
        ./cctag/Fitting.hpp
        ./cctag/ICCTag.hpp
        ./cctag/Identification.hpp
        ./cctag/ImageCut.hpp
        ./cctag/ImagePyramid.hpp
        ./cctag/Level.hpp
        ./cctag/Multiresolution.hpp
        ./cctag/Params.hpp
        ./cctag/Statistic.hpp
        ./cctag/SubPixEdgeOptimizer.hpp
        ./cctag/Types.hpp
        ./cctag/Vote.hpp
        ./cctag/algebra/matrix/Operation.hpp
        ./cctag/filter/cvRecode.hpp
        ./cctag/filter/thinning.hpp
        ./cctag/geometry/2DTransform.hpp
        ./cctag/geometry/Circle.hpp
        ./cctag/geometry/Distance.hpp
        ./cctag/geometry/Ellipse.hpp
        ./cctag/geometry/EllipseFromPoints.hpp
        ./cctag/geometry/Point.hpp
        ./cctag/optimization/conditioner.hpp
        ./cctag/utils/Accum.hpp
        ./cctag/utils/Debug.hpp
        ./cctag/utils/Defines.hpp
        ./cctag/utils/Exceptions.hpp
        ./cctag/utils/FileDebug.hpp
        ./cctag/utils/LogTime.hpp
        ./cctag/utils/Release.hpp
        ./cctag/utils/Singleton.hpp
        ./cctag/utils/Talk.hpp
        ./cctag/utils/VisualDebug.hpp
        ./cctag/utils/pcg_extras.hpp
        ./cctag/utils/pcg_random.hpp
        ./cctag/utils/pcg_uint128.hpp
        )

set(CUDA_cpp "")
set(CUDA_cu "")
set(CUDA_HEADERS "")

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

  set(CUDA_HEADERS
          ./cctag/cuda/assist.h
          ./cctag/cuda/cctag_cuda_runtime.h
          ./cctag/cuda/clamp.h
          ./cctag/cuda/cmp_list.h
          ./cctag/cuda/debug_image.h
          ./cctag/cuda/debug_is_on_edge.h
          ./cctag/cuda/debug_macros.hpp
          ./cctag/cuda/device_prop.hpp
          ./cctag/cuda/edge_list.h
          ./cctag/cuda/frame.h
          ./cctag/cuda/frame_07_vote.h
          ./cctag/cuda/framemeta.h
          ./cctag/cuda/frameparam.h
          ./cctag/cuda/geom_ellipse.h
          ./cctag/cuda/geom_matrix.h
          ./cctag/cuda/keep_time.hpp
          ./cctag/cuda/nearby_point.h
          ./cctag/cuda/onoff.h
          ./cctag/cuda/pinned_counters.h
          ./cctag/cuda/ptrstep.h
          ./cctag/cuda/recursive_sweep.h
          ./cctag/cuda/tag.h
          ./cctag/cuda/tag_cut.h
          ./cctag/cuda/tag_threads.h
          ./cctag/cuda/triple_point.h
          )


  set(CUDA_NVCC_FLAGS  "${CUDA_NVCC_FLAGS}")

  # this must go before CUDA_ADD_LIBRARY otherwise we won't be able to add it
  # after
  # https://github.com/Kitware/CMake/blob/master/Modules/FindCUDA.cmake#L147
  # INCLUDE_DIRECTORIES -- "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>"
  include_directories(${CMAKE_CURRENT_LIST_DIR}
          ${OpenCV_INCLUDE_DIRS} ${Eigen_INCLUDE_DIR}
          ${TBB_INCLUDE_DIRS}
          "${CMAKE_SOURCE_DIR}/src")

  CUDA_ADD_LIBRARY(CCTag ${CUDA_cpp} ${CUDA_cu} ${CCTag_cpp})

  # since we used CUDA_ADD_LIBRARY we cannot use PUBLIC or PRIVATE here
  target_link_libraries(CCTag
                       ${OpenCV_LIBS}
                       Boost::date_time Boost::chrono Boost::thread Boost::serialization Boost::system Boost::filesystem Boost::atomic Boost::program_options Boost::timer
                       Eigen3::Eigen
                       ${TBB_LIBRARIES} ${CUDA_CUDADEVRT_LIBRARY})

  if(NOT MSVC)
   target_link_libraries(CCTag pthread dl)
  endif()

  target_compile_definitions(CCTag
                         PUBLIC -DCCTAG_WITH_CUDA -DBOOST_NO_CXX11_CONSTEXPR
                         PRIVATE ${TBB_DEFINITIONS})

  if(CCTAG_HAVE_SHFL_DOWN_SYNC)
    target_compile_definitions(CCTag PRIVATE "-DCCTAG_HAVE_SHFL_DOWN_SYNC")
  endif()

  if(CCTAG_NO_THRUST_COPY_IF)
    target_compile_definitions(CCTag PRIVATE "-DCCTAG_NO_THRUST_COPY_IF")
  endif()


  # This is nececessary for the CCTagConfig.cmake to correctly export the
  # includes, always because we used CUDA_ADD_LIBRARY
  set_target_properties(CCTag PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
          "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>;$<BUILD_INTERFACE:${generated_dir}>;$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>;${CUDA_INCLUDE_DIRS};${Boost_INCLUDE_DIRS};${OpenCV_INCLUDE_DIRS}")
  # Just for testing
#  get_target_property(testprop CCTag INTERFACE_INCLUDE_DIRECTORIES )
#  message(STATUS "testprop: ${testprop}")

else() # without CUDA

  message( STATUS "Declare CCTag library" )
  add_library(CCTag ${CCTag_cpp})
  set_target_properties(CCTag PROPERTIES VERSION ${PROJECT_VERSION})

  target_include_directories(CCTag
                           PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>"
                                  "$<BUILD_INTERFACE:${generated_dir}>"
                                  "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>"
                           PUBLIC ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS}
                           PRIVATE ${TBB_INCLUDE_DIRS})

  # just for testing
#  get_target_property(testprop CCTag INTERFACE_INCLUDE_DIRECTORIES )
#  message(STATUS "testprop: ${testprop}")

  target_compile_definitions(CCTag
                             PRIVATE ${TBB_DEFINITIONS})

  target_link_libraries(CCTag
                        PUBLIC
                          ${OpenCV_LIBS}
                          Eigen3::Eigen
                          Boost::atomic Boost::chrono Boost::date_time Boost::filesystem Boost::serialization Boost::system Boost::thread Boost::timer
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
if(CCTAG_EIGEN_NO_ALIGN)
  target_compile_definitions(CCTag PUBLIC ${AV_EIGEN_DEFINITIONS})
endif()

set_target_properties(CCTag PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(CCTag PROPERTIES DEBUG_POSTFIX "d")


# EXPORTING THE LIBRARY
#
# place to put the cmake-related files
set(config_install_dir "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
# include directory for install
set(include_install_dir "${CMAKE_INSTALL_INCLUDEDIR}")
# the name for the generated config.hpp
set(cctag_config_header_name "${generated_dir}/cctag/cctag_config.hpp")
# the name for the generated header version file
set(cctag_version_header_name "${generated_dir}/cctag/version.hpp")
# where to install the generated files
set(install_dir_generated "${include_install_dir}/cctag")

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
configure_file("${PROJECT_SOURCE_DIR}/cmake/config.hpp.in" ${cctag_config_header_name} @ONLY)

# version file
configure_file("${PROJECT_SOURCE_DIR}/cmake/version.hpp.in" ${cctag_version_header_name} @ONLY)

# Targets:
#   * <prefix>/lib/libCCTag.a
#   * header location after install: <prefix>/include/
#   * headers can be included by C++ code `#include <popsift/popsift.h>`
install(TARGETS CCTag
        EXPORT "${targets_export_name}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        INCLUDES DESTINATION "${include_install_dir}")

# Headers:
include(cctagHelper)
cctag_install_headers(PUBLIC_HEADERS ${CCTag_HEADERS} ${CUDA_HEADERS}
        INSTALL_FOLDER ${include_install_dir})

# install the config and version generated files
install( FILES "${cctag_config_header_name}"
    DESTINATION "${install_dir_generated}")

install( FILES "${cctag_version_header_name}"
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

# Generate ${PROJECT_NAME}Targets.cmake in the build directory to use the library without installing it
export(TARGETS CCTag FILE "${generated_dir}/${targets_export_name}.cmake")

if(CCTAG_BUILD_APPS)
  add_subdirectory(applications)
endif()

if(CCTAG_BUILD_TESTS)
    add_subdirectory(cctag/test)
    add_subdirectory(cctag/geometry/test)
endif()
