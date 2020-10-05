#!/bin/bash

# assumes that we are at PyRep source dir
set -euxo pipefail

# python 3 install
cd $RLBENCH_ROOT
pip3 install -r requirements.txt
python3 setup.py install