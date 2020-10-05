#!/bin/bash

# assumes that we are at PyRep source dir
set -euxo pipefail

# these don't persist, set them in docker env
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VREP_ROOT
#export QT_QPA_PLATFORM_PLUGIN_PATH=$VREP_ROOT

# python 3 install
curl -o drake.tar.gz https://drake-packages.csail.mit.edu/drake/nightly/drake-latest-bionic.tar.gz
rm -rf /opt/drake
tar -xvzf drake.tar.gz -C /opt
rm -rf drake.tar.gz
yes "Y" | /opt/drake/share/drake/setup/install_prereqs
