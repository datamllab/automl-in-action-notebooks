#!/bin/bash
for filename in ipynb/*.ipynb; do
    target=$(echo $filename | cut -c 7-)
    target=${target%.*}".py"
    python tutobooks.py nb2py $filename py/$target
done
#for filename in py/*.py; do
    #target=$(echo $filename | cut -c 7-)
    #target=${target%.*}".py"
    #python tutobooks.py nb2py $filename py/$target
#done
