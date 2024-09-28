#!/bin/bash

# Set the URL for the dataset
url="https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"

# Download the zip file
echo "Downloading the dataset..."
wget $url -O DAVIS-2017-trainval-480p.zip

# Unzip the downloaded file
echo "Unzipping the dataset..."
unzip DAVIS-2017-trainval-480p.zip

# Remove the zip file after extraction
echo "Cleaning up..."
rm DAVIS-2017-trainval-480p.zip

echo "Done!"
