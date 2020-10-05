#!/bin/bash

# assumes that we are at PyRep source dir
set -euxo pipefail

# these don't persist, set them in docker env
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VREP_ROOT
#export QT_QPA_PLATFORM_PLUGIN_PATH=$VREP_ROOT

# python 3 install
cd $PYREP_ROOT
pip3 install -r requirements.txt
python3 setup.py install # install it globally