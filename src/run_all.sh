#!/bin/bash

for filename in py/*.py; do
    mkdir tmp
    cd tmp
    echo "Running $filename"
    if ipython "../$filename" > /dev/null; then
        echo "Succeeded $filename"
    else
        echo "Failed $filename"
    fi
    cd ..
    rm -rf tmp
done
