#
# Create an executable test from a source file. The associate executable has the same
# name as the source file.
#
# Usage:
#   add_boost_test(SOURCE <source>
#                [INCLUDE [items1...] ]
#                [LINK [items1...]
#                [PREFIX <string>])
#
# SOURCE is the source file contaning the test
# INCLUDE is an optional list of include directories
# LINK is an optional list of targets/libraries to link with
# PREFIX is an optional string to append to the name of the test (e.g. the module name)
#     For example if foo.cpp is passed as source, and "bar" as PREFIX the executable
#     will be named named "bar_foo". Without PREFIX it will be just "foo"
#
# The function parses each test file and extract the BOOST_AUTO_TEST_SUITEs so
# that each suite is run and visualised as a separated test. This means that no
# BOOST_AUTO_TEST_CASE must be left outside the suite or it will not be run during make test
#
# Function based on https://eb2.co/blog/2015/06/driving-boost-dot-test-with-cmake/
#
function(add_boost_test)

    set(options OPTIONS)
    set(oneValueArgs SOURCE PREFIX)
    set(multiValueArgs INCLUDE LINK)
    cmake_parse_arguments(param "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

#    message(STATUS "INCLUDE ${param_INCLUDE}")
#    message(STATUS "LINK ${param_LINK}")
#    message(STATUS "PREFIX ${param_PREFIX}")
#    message(STATUS "SOURCE ${param_SOURCE}")

    # get the name without the extension (NAME_WE)
    get_filename_component(TEST_EXECUTABLE_NAME ${param_SOURCE} NAME_WE)

    if(NOT ${param_PREFIX} STREQUAL "")
        set(TEST_EXECUTABLE_NAME "${param_PREFIX}_${TEST_EXECUTABLE_NAME}")
    endif()

    add_executable(${TEST_EXECUTABLE_NAME} ${param_SOURCE})

    if(NOT ${param_INCLUDE} STREQUAL "")
        target_include_directories(${TEST_EXECUTABLE_NAME} ${param_INCLUDE})
    endif()

    target_link_libraries(${TEST_EXECUTABLE_NAME}
            ${param_LINK} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

    # test_case must be always contained in a test_suite
    file(READ "${param_SOURCE}" SOURCE_FILE_CONTENTS)
    string(REGEX MATCHALL "BOOST_AUTO_TEST_SUITE\\( *([A-Za-z_0-9]+) *\\)"
            FOUND_TESTS ${SOURCE_FILE_CONTENTS})

    foreach(HIT ${FOUND_TESTS})
        string(REGEX REPLACE ".*\\( *([A-Za-z_0-9]+) *\\).*" "\\1" TEST_NAME ${HIT})

        add_test(NAME "${TEST_EXECUTABLE_NAME}.${TEST_NAME}"
                COMMAND ${TEST_EXECUTABLE_NAME}
                --run_test=${TEST_NAME} --catch_system_error=yes)
    endforeach()

#    add_test(NAME ${TEST_EXECUTABLE_NAME}
#             COMMAND ${TEST_EXECUTABLE_NAME} --catch_system_error=yes)

endfunction()