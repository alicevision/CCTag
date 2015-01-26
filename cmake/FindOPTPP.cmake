# Searches for an installation of the OPTPP library (http://software.sandia.gov/opt++/)
#
# Defines:
#
#   OPTPP_FOUND          True if OPTPP was found, else false
#   OPTPP_LIBRARIES      Libraries to link
#   OPTPP_INCLUDE_DIRS   The directories containing the header files
#   OPTPP_LDFLAGS        Extra linker flags
#
# To specify an additional directory to search, set OPTPP_ROOT.
#
# Author: Siddhartha Chaudhuri, 2011
#

set(OPTPP_FOUND FALSE)

if(NOT OPTPP_INCLUDE_DIRS)
  # Look for the OPTPP header, first in the user-specified location and then in the system locations
  set(OPTPP_INCLUDE_DOC "The directory containing the OPTPP include file Opt.h")
  find_path(OPTPP_INCLUDE_DIRS
            NAMES Opt.h
            PATHS ${OPTPP_ROOT}
            PATH_SUFFIXES "include" "include/OPTPP" "include/OPTPP" "include/opt++" "include/optpp"
            DOC ${OPTPP_INCLUDE_DOC}
            NO_DEFAULT_PATH)
message(WARNING, "heho: ${OPTPP_INCLUDE_DIRS}")
  if(NOT OPTPP_INCLUDE_DIRS)  # now look in system locations
    find_path(OPTPP_INCLUDE_DIRS NAMES Opt.h PATH_SUFFIXES "OPTPP" "OPTPP" "opt++" "optpp" DOC ${OPTPP_INCLUDE_DOC})
  endif()
endif()

# Only look for the library file in the immediate neighbourhood of the include directory
if(OPTPP_INCLUDE_DIRS)
  set(OPTPP_LIBRARY_DIRS ${OPTPP_INCLUDE_DIRS})

  if("${OPTPP_LIBRARY_DIRS}" MATCHES "/(OPT|opt)([+][+]|pp)$")
    # Strip off the trailing "/OPTPP" from the path
    GET_FILENAME_COMPONENT(OPTPP_LIBRARY_DIRS ${OPTPP_LIBRARY_DIRS} PATH)
  endif()

  if("${OPTPP_LIBRARY_DIRS}" MATCHES "/include$")
    # Strip off the trailing "/include" from the path
    GET_FILENAME_COMPONENT(OPTPP_LIBRARY_DIRS ${OPTPP_LIBRARY_DIRS} PATH)
  endif()

  # Look for libopt
  find_library(OPTPP_DEBUG_LIBRARY NAMES opt_d libopt_d optd liboptd PATH_SUFFIXES "" Debug
               PATHS ${OPTPP_LIBRARY_DIRS} ${OPTPP_LIBRARY_DIRS}/lib NO_DEFAULT_PATH)

  find_library(OPTPP_RELEASE_LIBRARY NAMES opt libopt PATH_SUFFIXES "" Release
               PATHS ${OPTPP_LIBRARY_DIRS} ${OPTPP_LIBRARY_DIRS}/lib NO_DEFAULT_PATH)

  set(OPTPP_LIBRARIES )
  if(OPTPP_DEBUG_LIBRARY AND OPTPP_RELEASE_LIBRARY)
    set(OPTPP_LIBRARIES debug ${OPTPP_DEBUG_LIBRARY} optimized ${OPTPP_RELEASE_LIBRARY})
  elseif(OPTPP_DEBUG_LIBRARY)
    set(OPTPP_LIBRARIES ${OPTPP_DEBUG_LIBRARY})
  elseif(OPTPP_RELEASE_LIBRARY)
    set(OPTPP_LIBRARIES ${OPTPP_RELEASE_LIBRARY})
  endif()

  # Look for libnewmat
  if(OPTPP_LIBRARIES)
    find_path(OPTPP_NEWMAT_INCLUDE_DIRS
            NAMES include.h
            PATHS ${OPTPP_ROOT}
            PATH_SUFFIXES "include" "include/newmat11" "include/newmat" "newmat11" "newmat"
            NO_DEFAULT_PATH)

    find_library(OPTPP_NEWMAT_DEBUG_LIBRARY NAMES newmat_d libnewmat_d newmatd libnewmatd PATH_SUFFIXES "" Debug
                 PATHS ${OPTPP_LIBRARY_DIRS} ${OPTPP_LIBRARY_DIRS}/lib NO_DEFAULT_PATH)
  
    find_library(OPTPP_NEWMAT_RELEASE_LIBRARY NAMES newmat libnewmat PATH_SUFFIXES "" Release
                 PATHS ${OPTPP_LIBRARY_DIRS} ${OPTPP_LIBRARY_DIRS}/lib NO_DEFAULT_PATH)

    set(OPTPP_INCLUDE_DIRS ${OPTPP_INCLUDE_DIRS} ${OPTPP_NEWMAT_INCLUDE_DIRS})
    if(OPTPP_NEWMAT_DEBUG_LIBRARY AND OPTPP_NEWMAT_RELEASE_LIBRARY)
      set(OPTPP_LIBRARIES ${OPTPP_LIBRARIES} debug ${OPTPP_NEWMAT_DEBUG_LIBRARY} optimized ${OPTPP_NEWMAT_RELEASE_LIBRARY})
    elseif(OPTPP_NEWMAT_DEBUG_LIBRARY)
      set(OPTPP_LIBRARIES ${OPTPP_LIBRARIES} ${OPTPP_NEWMAT_DEBUG_LIBRARY})
    elseif(OPTPP_NEWMAT_RELEASE_LIBRARY)
      set(OPTPP_LIBRARIES ${OPTPP_LIBRARIES} ${OPTPP_NEWMAT_RELEASE_LIBRARY})
    endif()
  endif()

  # Look for BLAS
  if(OPTPP_LIBRARIES)
    FIND_PACKAGE(BLAS)
    if(BLAS_FOUND)
      set(OPTPP_LIBRARIES ${OPTPP_LIBRARIES} ${BLAS_LIBRARIES})
      set(OPTPP_LDFLAGS ${BLAS_LINKER_FLAGS})
    else()
      message(STATUS "OPTPP: BLAS library not found")
      set(OPTPP_LIBRARIES )
    endif()
  endif()

message(WARNING, "heho: ${OPTPP_LIBRARIES}")
  if(OPTPP_LIBRARIES)
    set(OPTPP_FOUND TRUE)
  endif()
endif()

if(OPTPP_FOUND)
  if(NOT OPTPP_FIND_QUIETLY)
    message(STATUS "Found OPTPP: headers at ${OPTPP_INCLUDE_DIRS}, libraries at ${OPTPP_LIBRARIES}")
  endif()
else()
  if(OPTPP_FIND_REQUIRED)
    message(FATAL_ERROR "OPTPP not found")
  endif()
endif()

