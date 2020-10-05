#!/bin/bash

set -euxo pipefail


# required by vREP for video compression
apt install -y \
  libavcodec-dev \
  libavformat-dev \
  libswscale-dev

wget http://www.coppeliarobotics.com/files/V-REP_PRO_EDU_V3_6_2_Ubuntu16_04.tar.xz
tar -xJf V-REP_PRO_EDU_V3_6_2_Ubuntu16_04.tar.xz && rm -rf V-REP_PRO_EDU_V3_6_2_Ubuntu16_04.tar.xz
mv V-REP_PRO_EDU_V3_6_2_Ubuntu16_04 $VREP_ROOT
