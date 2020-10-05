#!/bin/bash

set -euxo pipefail

# torch 1.2, torchvision 0.4
# CUDA 10.0
# python2
# problem is numpy dependency upgrading to something not compatible with python2.7
# basically numpy>=1.17
# solution is to manually install all of torch and torchvision deps
# with a requirements.txt file
pip install \
  future \
  numpy==1.16.0 \
  pyyaml \
  requests \
  setuptools \
  six \
  typing
pip install --no-deps torch==1.2.0 torchvision==0.4.0 -f https://download.pytorch.org/whl/cu100/torch_stable.html

# do it for python3 virtualenv
#source $PYTHON37_VENV_ROOT/bin/activate
#pip install torch==1.2.0 torchvision==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
#deactivate

#apt-get -y install ipython ipython-notebook
#pip install \
#  jupyter \
#  opencv-python \
#  plyfile \
#  pandas \
#  tensorflow \
#  future \
#  typing \
#  open3d-python