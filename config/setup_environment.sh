#!/bin/bash
set -e

# assumes that the following environment variables have been set
# typically in entrypoint.sh if inside docker
#  export PYTHON_MODULES_ROOT=$HOME/code
#  export KEY_DYNAM_ROOT=$PYTHON_MODULES_ROOT/key_dynam
#  export DC_SOURCE_DIR=$PYTHON_MODULES_ROOT/pdc


function add_modules_to_pythonpath()
{
	export PYTHONPATH=$PYTHON_MODULES_ROOT:$PYTHONPATH
	export PYTHONPATH=$PYTHONPATH:$KEY_DYNAM_ROOT/external/contrastive-forward-model

	# make sure meshcat gets imported before the drake meshcat
	export PYTHONPATH=$PYTHONPATH:$MESHCAT_INSTALL_DIR
}

function use_drake(){
  export PYTHONPATH=${PYTHONPATH}:/opt/drake/lib/python3.6/site-packages
}

export -f add_modules_to_pythonpath
export -f use_drake

exec "$@"

