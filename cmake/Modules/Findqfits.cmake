# This module tries to find the LHAPDF installation on your system.

MESSAGE(STATUS "Looking for qfits...")

FIND_LIBRARY(QFITS_LIBRARIES
  NAMES
  qfits
  PATHS
  $ENV{QFITS_PATH}/lib
  /usr/lib
  /usr/local/lib
  )
MESSAGE(STATUS "HALLO $ENV{QFITS_PATH} uu ${QFITS_LIBRARIES}" )


FIND_PATH(QFITS_INCLUDE_DIR
 qfits.h
 $ENV{QFITS_PATH}/include
 /usr/include
 /usr/local/include
 )
MESSAGE(STATUS "HALLO $ENV{QFITS_PATH} uu ${QFITS_INCLUDE}" )
