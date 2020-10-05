#!/bin/bash

set -euxo pipefail

apt-get update
apt install -y --no-install-recommends \
  python-pip \
  python-dev \
  terminator \
  tmux \
  vim \
  gedit \
  git \
  openssh-client \
  unzip \
  htop \
  libopenni-dev \
  apt-utils \
  usbutils \
  dialog \
  ffmpeg \
  nvidia-settings \
  cmake-curses-gui \
  libyaml-dev \
  virtualenv \
  wget \
  python3-tk \
  curl \
  dbus-x11 \
  libqt5gui5 # needed for vREP libqt5gui5
  # mesa-utils # this will give us glxgears

#apt install -y dbus-x11 # needed for terminator fonts


