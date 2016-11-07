# IF(COMMAND cmake_policy)
cmake_policy(SET CMP0054 OLD) # behaviour keywords in CMake if tests
# ENDIF(COMMAND cmake_policy)

#file( GLOB_RECURSE
#      CCTag_hpp
#      ./cctag/*.hpp
#)

file( GLOB_RECURSE
      CCTag_cpp
      ./cctag/*.cpp
)

if(USE_CUDA)
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
    message( FATAL_ERROR "We must link CCTag library static with CUDA" )
  endif()

  set(CUDA_NVCC_FLAGS  "${CUDA_NVCC_FLAGS};-DSM_ARCH=100;-DSM=100;-DCUB_CDP")
 
  # this must go before CUDA_ADD_LIBRARY otherwise we won't be able to add it
  # after
  # https://github.com/Kitware/CMake/blob/master/Modules/FindCUDA.cmake#L147
  INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${Eigen_INCLUDE_DIR}
                      "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>"
                      "${CMAKE_SOURCE_DIR}/3rdparty/Cuda-7.0-cub")
 
  CUDA_ADD_LIBRARY( CCTag STATIC ${CUDA_cpp} ${CUDA_cu} ${CCTag_cpp})

  # since we used CUDA_ADD_LIBRARY we cannot use PUBLIC or PRIVATE here
  target_link_libraries(CCTag 
                       ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Eigen_LIBRARIES} 
                       ${TBB_LIBRARIES} ${PNG_LIBRARIES} ${JPEG_LIBRARIES} 
                       ${CUDA_CUDADEVRT_LIBRARY} pthread dl)

  target_compile_definitions(CCTag 
                         PUBLIC ${Boost_DEFINITIONS} 
                         PUBLIC -DWITH_CUDA -DSM_ARCH=100 -DSM=100 -DCUB_CDP
                         PRIVATE ${TBB_DEFINITIONS} ${PNG_DEFINITIONS}) 

  # enable serialization
  if(CCTAG_SERIALIZE)
    target_compile_definitions(CCTag PUBLIC "-DCCTAG_SERIALIZE")
  endif(CCTAG_SERIALIZE)
  # Disable output stream
  if(CCTAG_NO_COUT)
    target_compile_definitions(CCTag PUBLIC "-DCCTAG_NO_COUT")
  endif(CCTAG_NO_COUT)

  # This is nececessary for the CCTagConfig.cmake to correctly export the
  # includes, always because we used CUDA_ADD_LIBRARY
  set_target_properties(CCTag PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIRS};${Boost_INCLUDE_DIRS};${Eigen_INCLUDE_DIR};${OpenCV_INCLUDE_DIRS}")
  get_target_property(testprop CCTag INTERFACE_INCLUDE_DIRECTORIES )
  message(STATUS "testprop: ${testprop}")

else(USE_CUDA)

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
                           PUBLIC ${Boost_INCLUDE_DIRS} ${Eigen_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS}
                           PRIVATE ${TBB_INCLUDE_DIR} ${PNG_INCLUDE_DIRS} ${JPEG_INCLUDE_DIR}
                           PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>"
                                  "$<INSTALL_INTERFACE:include/>") 

  get_target_property(testprop CCTag INTERFACE_INCLUDE_DIRECTORIES )
  message(STATUS "testprop: ${testprop}")

  target_compile_definitions(CCTag 
                           PUBLIC ${Boost_DEFINITIONS} 
                           PRIVATE ${TBB_DEFINITIONS} ${PNG_DEFINITIONS}) 
  
  # enable serialization
  if(CCTAG_SERIALIZE)
    target_compile_definitions(CCTag PUBLIC "-DCCTAG_SERIALIZE")
  endif(CCTAG_SERIALIZE)
  
  # Disable output stream
  if(CCTAG_NO_COUT)
    target_compile_definitions(CCTag PUBLIC "-DCCTAG_NO_COUT")
  endif(CCTAG_NO_COUT)
  
  target_link_libraries(CCTag 
                      PUBLIC ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Eigen_LIBRARIES} 
                      PRIVATE ${TBB_LIBRARIES} ${PNG_LIBRARIES} ${JPEG_LIBRARIES} pthread dl)

endif(USE_CUDA)



# EXPORTING THE LIBRARY
#
# place to put the cmake-related files
set(config_install_dir "lib/cmake/${PROJECT_NAME}")
# include directory for install
set(include_install_dir "include")

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
if(USE_CUDA)
  install(DIRECTORY "cuda"
          DESTINATION "${include_install_dir}"
          FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")
endif()

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