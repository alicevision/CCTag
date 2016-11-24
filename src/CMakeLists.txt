# IF(COMMAND cmake_policy)
cmake_policy(SET CMP0054 OLD) # behaviour keywords in CMake if tests
# ENDIF(COMMAND cmake_policy)

file( GLOB_RECURSE
      CCTag_cpp
      ./cctag/*.cpp
)

if(WITH_CUDA)
  file( GLOB_RECURSE
        CUDA_cpp
        ./cuda/*.cpp
  )
  file( GLOB_RECURSE
        CUDA_cu
        ./cuda/*.cu
  )

  # add_definitions("-DSM_TARGETS=-gencode=arch=compute_52,code=\"sm_52,compute_52\"")

  # set(CUDA_SEPARABLE_COMPILATION ON)
  # set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
  # set(BUILD_SHARED_LIBS ON)

  if(NOT COMPILE_STATIC_CCTAG_LIBRARY)
    message( FATAL_ERROR "In order to use CUDA the library must be build as static library. \
             To disable cuda and build a dynamic library use -DWITH_CUDA=OFF" )
  endif()

  set(CUDA_NVCC_FLAGS  "${CUDA_NVCC_FLAGS};-DSM_ARCH=${CCTAG_SM_ARCH};-DSM=${CCTAG_SM};-DCUB_CDP")
 
  # this must go before CUDA_ADD_LIBRARY otherwise we won't be able to add it
  # after
  # https://github.com/Kitware/CMake/blob/master/Modules/FindCUDA.cmake#L147
  INCLUDE_DIRECTORIES("$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>" 
                      ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${Eigen_INCLUDE_DIR}
                      "${CMAKE_SOURCE_DIR}/3rdparty/Cuda-7.0-cub")
 
  CUDA_ADD_LIBRARY( CCTag STATIC ${CUDA_cpp} ${CUDA_cu} ${CCTag_cpp})

  # since we used CUDA_ADD_LIBRARY we cannot use PUBLIC or PRIVATE here
  target_link_libraries(CCTag 
                       ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Eigen_LIBRARIES} 
                       ${TBB_LIBRARIES} ${PNG_LIBRARIES} ${JPEG_LIBRARIES} 
                       ${CUDA_CUDADEVRT_LIBRARY} pthread dl)

  target_compile_definitions(CCTag 
                         PUBLIC ${Boost_DEFINITIONS} 
                         PUBLIC -DWITH_CUDA -DSM_ARCH=${CCTAG_SM_ARCH} -DSM=${CCTAG_SM} -DCUB_CDP
                         PRIVATE ${TBB_DEFINITIONS} ${PNG_DEFINITIONS}) 

  # This is nececessary for the CCTagConfig.cmake to correctly export the
  # includes, always because we used CUDA_ADD_LIBRARY
  set_target_properties(CCTag PROPERTIES INTERFACE_INCLUDE_DIRECTORIES 
       "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>;${CUDA_INCLUDE_DIRS};${Boost_INCLUDE_DIRS};${Eigen_INCLUDE_DIR};${OpenCV_INCLUDE_DIRS}")
  get_target_property(testprop CCTag INTERFACE_INCLUDE_DIRECTORIES )
  message(STATUS "testprop: ${testprop}")

else(WITH_CUDA)

  set(CUDA_cpp "")

  if( COMPILE_STATIC_CCTAG_LIBRARY )
    message( STATUS "Compiling static library" )
    add_library( CCTag STATIC ${CCTag_cpp} )
  else( COMPILE_STATIC_CCTAG_LIBRARY )
    message( STATUS "Compiling shared library" )
    add_library( CCTag SHARED ${CCTag_cpp} )
    set_target_properties(CCTag PROPERTIES VERSION ${PROJECT_VERSION})
  endif( COMPILE_STATIC_CCTAG_LIBRARY )

  target_include_directories(CCTag 
                           PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>"
                                  "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>"
                           PUBLIC ${Boost_INCLUDE_DIRS} ${Eigen_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS}
                           PRIVATE ${TBB_INCLUDE_DIR} ${PNG_INCLUDE_DIRS} ${JPEG_INCLUDE_DIR}) 

  get_target_property(testprop CCTag INTERFACE_INCLUDE_DIRECTORIES )
  message(STATUS "testprop: ${testprop}")

  target_compile_definitions(CCTag 
                           PUBLIC ${Boost_DEFINITIONS} 
                           PRIVATE ${TBB_DEFINITIONS} ${PNG_DEFINITIONS}) 
  
  target_link_libraries(CCTag 
                      PUBLIC ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Eigen_LIBRARIES} 
                      PRIVATE ${TBB_LIBRARIES} ${PNG_LIBRARIES} ${JPEG_LIBRARIES} pthread dl)

endif(WITH_CUDA)

# enable serialization
if(CCTAG_SERIALIZE)
  target_compile_definitions(CCTag PUBLIC "-DCCTAG_SERIALIZE")
endif(CCTAG_SERIALIZE)
# Disable output stream
if(CCTAG_NO_COUT)
  target_compile_definitions(CCTag PUBLIC "-DCCTAG_NO_COUT")
endif(CCTAG_NO_COUT)
# Enable visual debug
if(VISUAL_DEBUG)
  target_compile_definitions(CCTag PRIVATE "-DVISUAL_DEBUG")
endif(VISUAL_DEBUG)


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
install(DIRECTORY "cctag"
        DESTINATION "${include_install_dir}"
        FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")
if(WITH_CUDA)
  install(DIRECTORY "cuda"
          DESTINATION "${include_install_dir}"
          FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")
endif()

install( FILES "${generated_dir}/${confighpp_name}"
    DESTINATION "${include_install_dir}")

install( FILES "${generated_dir}/${versionhpp_name}"
    DESTINATION "${include_install_dir}")

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

add_subdirectory(applications)