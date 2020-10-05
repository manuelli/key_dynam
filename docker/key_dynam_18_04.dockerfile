FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ARG USER_NAME
ARG USER_PASSWORD
ARG USER_ID
ARG USER_GID

# https://askubuntu.com/questions/909277/avoiding-user-interaction-with-tzdata-when-installing-certbot-in-a-docker-contai
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt install sudo
RUN useradd -ms /bin/bash $USER_NAME
RUN usermod -aG sudo $USER_NAME
RUN yes $USER_PASSWORD | passwd $USER_NAME

# set uid and gid to match those outside the container
RUN usermod -u $USER_ID $USER_NAME
RUN groupmod -g $USER_GID $USER_NAME

WORKDIR /home/$USER_NAME
ENV USER_HOME_DIR=/home/$USER_NAME
ARG KEY_DYNAM_ROOT=key_dynam

# install python3
# copied from https://github.com/FNNDSC/ubuntu-python3/blob/master/Dockerfile
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && pip3 install --upgrade pip

# install pytorch
RUN pip3 install torch==1.2.0 torchvision==0.4.0 -f https://download.pytorch.org/whl/cu100/torch_stable.html

# install other dependencies
COPY key_dynam/docker/install_dependencies.sh /tmp/install_dependencies.sh
RUN /tmp/install_dependencies.sh

# key_dynam requirements
COPY $KEY_DYNAM_ROOT/requirements.txt /tmp/key_dynam_requirements.txt
RUN pip3 install -r /tmp/key_dynam_requirements.txt

# install VREP
# RUN mkdir -p $USER_HOME_DIR/external
# ENV VREP_ROOT=$USER_HOME_DIR/external/VREP
# COPY $KEY_DYNAM_ROOT/docker/install_vrep.sh /tmp/install_vrep.sh
# RUN /tmp/install_vrep.sh
#
# # install PyRep and it's dependencies
# ENV PYREP_ROOT=$USER_HOME_DIR/software/PyRep
# RUN mkdir -p $PYREP_ROOT
# COPY PyRep/ $PYREP_ROOT
# COPY $KEY_DYNAM_ROOT/docker/install_PyRep.sh /tmp/install_PyRep.sh
# RUN /tmp/install_PyRep.sh
#
# # install RLbench dependencies, only in python3 virtualenv
# ENV RLBENCH_ROOT=$USER_HOME_DIR/software/PyRep
# RUN mkdir -p $RLBENCH_ROOT
# COPY RLBench/ $RLBENCH_ROOT


# install drake
COPY $KEY_DYNAM_ROOT/docker/install_drake.sh /tmp/install_drake.sh
# RUN /tmp/install_drake.sh
# a way to proceed without errors
RUN /tmp/install_drake.sh | tee docker_build.log || echo "Drake install failed!"


# install meshcat-python dependencies
ENV MESHCAT_INSTALL_DIR=/usr/local/lib/python3.6/dist-packages/meshcat-0.0.18-py3.6.egg
RUN mkdir -p $USER_HOME_DIR/software && cd $USER_HOME_DIR/software
COPY $KEY_DYNAM_ROOT/docker/install_meshcat.sh /tmp/install_meshcat.sh
RUN /tmp/install_meshcat.sh



# change color of terminator inside docker
RUN mkdir -p .config/terminator
COPY $KEY_DYNAM_ROOT/docker/terminator_config .config/terminator/config


# change ownership of everything to our user
RUN cd ${USER_HOME_DIR} && echo $(pwd) && chown $USER_NAME:$USER_NAME -R .


ENTRYPOINT bash -c "source ~/code/key_dynam/docker/entrypoint.sh && /bin/bash"

