#!/bin/bash
set -e
CURRDIR="$( cd "$( dirname "$( readlink -f "${BASH_SOURCE[0]}" )" )" && pwd )"
. "${CURRDIR}/env.sh"


# downloadFromAliceVisionDependencies TARGET_FULL_NAME INSTALL_PATH
downloadFromAliceVisionDependencies()
{
    download_files_from_tar "https://github.com/alicevision/AliceVisionDependencies/releases/download/$1/$1.tgz" $2
    return 0
}

set -x

downloadFromAliceVisionDependencies boost-1.70.0 ${DEPS_INSTALL_PATH}
downloadFromAliceVisionDependencies eigen-3.3.7 ${DEPS_INSTALL_PATH}
downloadFromAliceVisionDependencies opencv-3.4.2 ${DEPS_INSTALL_PATH}
