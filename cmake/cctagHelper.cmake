#
# Function for installing the library headers preserving the (sub)-folder structure.
# Apparently this cannot be achieved by setting PUBLIC_HEADERS (https://gitlab.kitware.com/cmake/cmake/issues/17790)
#
# Usage:
#   cctag_install_headers(PUBLIC_HEADERS [item1...]
#                INSTALL_FOLDER export_name)
#
# PUBLIC_HEADERS The list of headers.
# INSTALL_FOLDER The install folder.
#
function(cctag_install_headers)
    set(options "")
    set(singleValues INSTALL_FOLDER)
    set(multipleValues PUBLIC_HEADERS)
    cmake_parse_arguments(INSTALLHEADERS "${options}" "${singleValues}" "${multipleValues}" ${ARGN})

    foreach(header ${INSTALLHEADERS_PUBLIC_HEADERS})
        get_filename_component(dir ${header} DIRECTORY)
#        message("${dir} -- ${header} -- ${INSTALLHEADERS_INSTALL_FOLDER}")
        install(FILES ${header} DESTINATION ${INSTALLHEADERS_INSTALL_FOLDER}/${dir})
    endforeach(header)

endfunction()