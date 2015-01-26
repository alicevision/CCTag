find_path(UBlas_INCLUDE_DIR boost/numeric/ublas)

if(UBlas_INCLUDE_DIR)
   set(UBlas_FOUND TRUE)
   set(UBlas_DEFINITIONS -DWITH_UBlas)
   message(STATUS "Found UBlas: ${UBlas_INCLUDE_DIR}")
endif()

