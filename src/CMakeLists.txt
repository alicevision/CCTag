FILE(
	GLOB_RECURSE
	CCTag_hpp
	./cctag/*.hpp
)

FILE(
	GLOB_RECURSE
	CCTag_cpp
	./cctag/*.cpp
)

set(CCTagDetect_cpp ./cctag/detection/main.cpp)

add_library( CCTag SHARED ${CCTag_cpp})
target_link_libraries( CCTag PUBLIC
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    ${OPTPP_LIBRARIES}
    ${Ceres_LIBRARY}
    ${Eigen_LIBRARIES}
    ${PNG_LIBRARIES}
    ${JPEG_LIBRARIES}
    ${Eigen_LIBRARIES}
    ${Gomp_LIBRARIES}
    ${GLOG_LIBRARIES}
    ${GSL_LIBRARIES}
    ${LAPACK_LIBRARIES})

target_include_directories( CCTag PUBLIC .)

add_executable( detection ${CCTagDetect_cpp})
target_link_libraries( detection CCTag)

install(TARGETS CCTag LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(FILES ${CCTag_hpp} DESTINATION include/cctag)

