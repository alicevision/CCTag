#!/bin/bash
rm -rf build
mkdir build
cd build
for branch in $(git for-each-ref --format='%(refname:short)' refs/heads/); do
    echo $branch
    mkdir $branch
    cd $branch
    git checkout $branch
    cmake ../..
    make -j6
    ./src/detection ../../sample/00001.png ../../cctagLibraries/4Crowns/ids.txt ../../parameters/param4.xml
    cd ..
done
cd ..
