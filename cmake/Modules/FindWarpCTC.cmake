# Try to find Warp-CTC from Baidu
# Once found these will be defined
# WARPCTC_FOUND
# WARPCTC_INCLUDE_DIRS
# WARPCTC_LIBRARIES

set (WARPCTC_ROOT $ENV{WARPCTC_ROOT}/)

find_path(WARPCTC_INCLUDE_DIR
        NAMES ctc.h
        PATHS ${WARPCTC_ROOT}/include
        DOC "Warp-CTC include directory"
)

find_library(WARPCTC_LIBRARY
        NAMES warpctc
        PATHS ${WARPCTC_ROOT}/build
        DOC "Warp-CTC library"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(WARPCTC DEFAULT_MSG WARPCTC_INCLUDE_DIR WARPCTC_LIBRARY)

if (WAPRCTC_FOUND)
    set(WARPCTC_INCLUDE_DIRS ${WARPCTC_INCLUDE_DIR})
    set(WARPCTC_LIBRARIES ${WARPCTC_LIBRARY})
endif()

mark_as_advanced(WAPRCTC_ROOT WARPCTC_INCLUDE_DIR WARPCTC_LIBRARY)