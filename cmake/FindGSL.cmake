# Script found on KDE-edu list
# Permission obtained from Jan Woetzel to use under a BSD-style license.
#
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#
# Look for the header file
# Try to find gnu scientific library GSL
# See
# http://www.gnu.org/software/gsl/  and
# http://gnuwin32.sourceforge.net/packages/gsl.htm
#
# Once run this will define:
#
# GSL_FOUND       = system has GSL lib
#
# GSL_LIBRARIES   = full path to the libraries
#    on Unix/Linux with additional linker flags from "gsl-config --libs"
#
# CMAKE_GSL_CXX_FLAGS  = Unix compiler flags for GSL, essentially "`gsl-config --cxxflags`"
#
# GSL_INCLUDE_DIR      = where to find headers
#
# GSL_LINK_DIRECTORIES = link directories, useful for rpath on Unix
# GSL_EXE_LINKER_FLAGS = rpath on Unix
#
# Felix Woelk 07/2004
# Jan Woetzel
#
# www.mip.informatik.uni-kiel.de
# --------------------------------

# JW tested with gsl-1.8, Windows XP, MSVS 7.1
SET(GSL_POSSIBLE_ROOT_DIRS
  ${GSL_ROOT_DIR}
  $ENV{GSL_ROOT_DIR}
  ${GSL_DIR}
  ${GSL_HOME}
  $ENV{GSL_DIR}
  $ENV{GSL_HOME}
  $ENV{EXTRA}
  )
FIND_PATH(GSL_INCLUDE_DIR
  NAMES gsl/gsl_cdf.h gsl/gsl_randist.h
  PATHS ${GSL_POSSIBLE_ROOT_DIRS}
  PATH_SUFFIXES include
  DOC "GSL header include dir"
  )

FIND_LIBRARY(GSL_GSL_LIBRARY
  NAMES gsl libgsl
  PATHS  ${GSL_POSSIBLE_ROOT_DIRS}
  PATH_SUFFIXES lib
  DOC "GSL library dir" )

FIND_LIBRARY(GSL_GSLCBLAS_LIBRARY
  NAMES gslcblas libgslcblas
  PATHS  ${GSL_POSSIBLE_ROOT_DIRS}
  PATH_SUFFIXES lib
  DOC "GSL cblas library dir" )

SET(GSL_LIBRARIES ${GSL_GSL_LIBRARY} ${GSL_GSLCBLAS_LIBRARY})

#MESSAGE("DBG\n"
#  "GSL_GSL_LIBRARY=${GSL_GSL_LIBRARY}\n"
#  "GSL_GSLCBLAS_LIBRARY=${GSL_GSLCBLAS_LIBRARY}\n"
#  "GSL_LIBRARIES=${GSL_LIBRARIES}")



IF(GSL_LIBRARIES)
  IF(GSL_INCLUDE_DIR OR GSL_CXX_FLAGS)
    SET(GSL_FOUND 1)
  ENDIF(GSL_INCLUDE_DIR OR GSL_CXX_FLAGS)
ELSE(GSL_LIBRARIES)
  IF (GSL_FIND_REQUIRED)
    message(SEND_ERROR "FindGSL.cmake: Unable to find the required GSL libraries")
  ENDIF(GSL_FIND_REQUIRED)
ENDIF(GSL_LIBRARIES)

