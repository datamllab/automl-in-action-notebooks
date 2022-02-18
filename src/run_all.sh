#!/bin/bash

mkdir tmp
cd tmp
for filename in ../py/*.py; do
    echo "Running $filename"
    if ipython "$filename" > /dev/null; then
        echo "Succeeded $filename"
    else
        echo "Failed $filename"
    fi
done
cd ..
rm -rf tmp
