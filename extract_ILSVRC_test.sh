#!/bin/bash
#
# Create train directory; move .tar file; change directory
mkdir imagenet/test && mv ILSVRC2012_img_test.tar imagenet/test/ && cd imagenet/test
# Extract training set; remove compressed file
tar -xvf ILSVRC2012_img_test.tar  
# tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
#
# At this stage imagenet/train will contain 1000 compressed .tar files, one for each category
#
# For each .tar file: 
#   1. create directory with same name as .tar file
#   2. extract and copy contents of .tar file into directory
#   3. remove .tar file
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
#
# This results in a training directory like so:
#
#  imagenet/train/
#  ├── n01440764
#  │   ├── n01440764_10026.JPEG
#  │   ├── n01440764_10027.JPEG
#  │   ├── ......
#  ├── ......
#
# Change back to original directory
#
# Extract the validation data and move images to subfolders:
#
# Create validation directory; move .tar file; change directory; extract validation .tar; remove compressed file
# Check total files after extract
#
#  $ find train/ -name "*.JPEG" | wc -l
#  1281167
#  $ find val/ -name "*.JPEG" | wc -l
#  50000
#
