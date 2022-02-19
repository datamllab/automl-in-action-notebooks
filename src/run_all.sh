#!/bin/bash
wget https://github.com/datamllab/automl-in-action-notebooks/raw/master/data/mnist.tar.gz
tar xzf mnist.tar.gz

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
