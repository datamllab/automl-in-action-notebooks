#!/bin/bash
#for filename in *.ipynb; do
    #jupyter nbconvert "$filename" --to script
#done

for filename in *.py; do
    echo "Running $filename"
    if ipython "$filename" > /dev/null; then
        echo "Succeeded $filename"
    else
        echo "Failed $filename"
    fi
done
