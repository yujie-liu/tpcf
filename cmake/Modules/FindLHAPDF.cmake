# This module tries to find the LHAPDF installation on your system.

MESSAGE(STATUS "Looking for LHAPDF...")


# try to find LHAPDF in user defined path
FIND_LIBRARY(LHAPDF_LIB
  NAMES
  LHAPDF
  PATHS
  ${LHAPDF_PREFIX}/lib
  )


# if not try to find LHAPDF in standard instalation paths
IF(${LHAPDF_LIB} MATCHES "LHAPDF_LIB-NOTFOUND")
  FIND_LIBRARY(LHAPDF_LIB
    NAMES
    LHAPDF
    PATHS
    /usr/lib
    /usr/local/lib
    )
ENDIF(${LHAPDF_LIB} MATCHES "LHAPDF_LIB-NOTFOUND")

# if not found try to use lhapdf-config tool
IF(${LHAPDF_LIB} MATCHES "LHAPDF_LIB-NOTFOUND")
  FIND_PROGRAM(LHAPDF_CONFIG_EXECUTABLE NAMES lhapdf-config)
  IF(${LHAPDF_CONFIG_EXECUTABLE} MATCHES "LHAPDF_CONFIG_EXECUTABLE-NOTFOUND")
    MESSAGE(STATUS "Looking for LHAPDF... - lhapdf-config executable not found")
  ELSE(${LHAPDF_CONFIG_EXECUTABLE} MATCHES "LHAPDF_CONFIG_EXECUTABLE-NOTFOUND")
    MESSAGE(STATUS "Looking for LHAPDF... - using lhapdf-config executable")
    EXEC_PROGRAM(${LHAPDF_CONFIG_EXECUTABLE} ARGS "--prefix" OUTPUT_VARIABLE LHAPDF_PREFIX)
    FIND_LIBRARY(LHAPDF_LIB
      NAMES
      LHAPDF
      PATHS
      ${LHAPDF_PREFIX}/lib
      )
  ENDIF(${LHAPDF_CONFIG_EXECUTABLE} MATCHES "LHAPDF_CONFIG_EXECUTABLE-NOTFOUND")
ENDIF(${LHAPDF_LIB} MATCHES "LHAPDF_LIB-NOTFOUND")

IF(NOT ${LHAPDF_LIB} MATCHES "LHAPDF_LIB-NOTFOUND")
  FIND_PATH(LHAPDF_INCLUDE
    LHAPDF/LHAPDF.h
    /usr/include
    /usr/local/include
    ${LHAPDF_PREFIX}/include
    )
ENDIF(NOT ${LHAPDF_LIB} MATCHES "LHAPDF_LIB-NOTFOUND")



# final printout.
IF((${LHAPDF_LIB} MATCHES "LHAPDF_LIB-NOTFOUND") OR (${LHAPDF_INCLUDE} MATCHES "LHAPDF_INCLUDE-NOTFOUND"))
  MESSAGE( FATAL_ERROR "!!!!! LHAPDF library not found.          !!!!!\n"
    "!!!!! If it is installed                 !!!!!\n"
    "!!!!! in custom location, please edit    !!!!!\n"
    "!!!!! config/build.properties file       !!!!!")
ELSE((${LHAPDF_LIB} MATCHES "LHAPDF_LIB-NOTFOUND") OR (${LHAPDF_INCLUDE} MATCHES "LHAPDF_INCLUDE-NOTFOUND"))
  MESSAGE(STATUS "Looking for LHAPDF... - found " ${LHAPDF_LIB}  )
  MESSAGE(STATUS "Looking for LHAPDF... - found " ${LHAPDF_INCLUDE} )
  include_directories(${LHAPDF_INCLUDE})
ENDIF((${LHAPDF_LIB} MATCHES "LHAPDF_LIB-NOTFOUND") OR (${LHAPDF_INCLUDE} MATCHES "LHAPDF_INCLUDE-NOTFOUND"))
