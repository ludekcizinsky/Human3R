#!/bin/bash

# Download CUT3R model
cd src
gdown --fuzzy https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link
cd ..

# Download MultiHMR model
mkdir -p src/models/multihmr
wget https://download.europe.naverlabs.com/ComputerVision/MultiHMR/multiHMR_896_L.pt \
    -O './src/models/multihmr/multiHMR_896_L.pt' \
    --no-check-certificate \
    --continue