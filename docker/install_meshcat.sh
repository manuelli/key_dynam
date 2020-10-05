#!/bin/bash

# assumes that we are at PyRep source dir
set -euxo pipefail

pip3 install tornado pyzmq

git clone https://github.com/rdeits/meshcat-python.git
cd meshcat-python
python3 setup.py install