# - Find Ceres library
# Find the native Ceres includes and library
# This module defines
#  Ceres_INCLUDE_DIRS, where to find ceres.h, Set when
#                      Ceres_INCLUDE_DIR is found.
#  Ceres_LIBRARIES, libraries to link against to use Ceres.
#  Ceres_ROOT_DIR, The base directory to search for Ceres.
#                  This can also be an environment variable.
#  Ceres_FOUND, If false, do not try to use Ceres.
#
# also defined, but not for general use are
#  Ceres_LIBRARY, where to find the Ceres library.

# If Ceres_ROOT_DIR was defined in the environment, use it.
IF(NOT Ceres_ROOT_DIR AND NOT $ENV{Ceres_ROOT_DIR} STREQUAL "")
  SET(Ceres_ROOT_DIR $ENV{Ceres_ROOT_DIR})
ENDIF()

SET(_ceres_SEARCH_DIRS
  ${Ceres_ROOT_DIR}
  /usr/local
  /sw # Fink
  /opt/local # DarwinPorts
  /opt/csw # Blastwave
  /opt/lib/ceres
)

FIND_PATH(Ceres_INCLUDE_DIR
  NAMES
    ceres/ceres.h
  HINTS
    ${_ceres_SEARCH_DIRS}
  PATH_SUFFIXES
    include
)
# TODO: Is Ceres_CONFIG_INCLUDE_DIR really needed? Or is it by default when the installation is correct?
FIND_PATH(Ceres_CONFIG_INCLUDE_DIR
  NAMES
    ceres/internal/config.h
  HINTS
    ${_ceres_SEARCH_DIRS}
  PATH_SUFFIXES
    config
)
set(Ceres_INCLUDE_DIR ${Ceres_INCLUDE_DIR} ${Ceres_CONFIG_INCLUDE_DIR})

message( WARNING "est-ce que tu m'entends he ho - dixit TRAGEDY ${Ceres_INCLUDE_DIR}")
FIND_LIBRARY(Ceres_LIBRARY
  NAMES
    ceres
  HINTS
    ${_ceres_SEARCH_DIRS}
  PATH_SUFFIXES
    lib64 lib
  )

# handle the QUIETLY and REQUIRED arguments and set Ceres_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(ceres DEFAULT_MSG
    Ceres_LIBRARY Ceres_INCLUDE_DIR)

IF(Ceres_FOUND)
  SET(Ceres_LIBRARIES ${Ceres_LIBRARY})
  SET(Ceres_INCLUDE_DIRS ${Ceres_INCLUDE_DIR})
ENDIF(Ceres_FOUND)

MARK_AS_ADVANCED(
  Ceres_INCLUDE_DIR
  Ceres_LIBRARY
)
