#!/bin/bash
set -e

export PYTHON_MODULES_ROOT=$HOME/code

# setup for key_dynam
export KEY_DYNAM_ROOT=$PYTHON_MODULES_ROOT/key_dynam
source $KEY_DYNAM_ROOT/config/setup_environment.sh

# setup for pdc
# comment this out if you don't have pdc cloned in the workspace
export DC_SOURCE_DIR=$PYTHON_MODULES_ROOT/pdc
source $DC_SOURCE_DIR/config/setup_environment.sh

export DATA_ROOT=$HOME/data
export DATA_SSD_ROOT=$HOME/data_ssd
export DC_DATA_DIR=$HOME/data

exec "$@"

cd ~/code
