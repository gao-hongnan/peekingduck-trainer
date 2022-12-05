#!/bin/bash
# Download CIFAR-10 Dataset and extract it
# Example usage: bash scripts/download_cifar10.sh
# parent
# ├── src
# └── data
#     └── cifar10  ← downloads here

# Download/unzip images and labels
datasets_path='./data' # unzip directory
dataset_name='cifar10'
url=https://github.com/gao-hongnan/peekingduck-trainer/releases/download/v0.0.1-alpha/cifar10.zip
zip_file='cifar10.zip' # or 'coco128-segments.zip', 68 MB
echo 'Downloading' $url$zip_file ' ...'
mkdir -p $d
wget -P $datasets_path/$dataset_name $url
unzip $datasets_path/$dataset_name/$zip_file -d $datasets_path
rm $datasets_path/$dataset_name/$zip_file

wait # finish background tasks