#!/bin/bash

# enable module support
source /etc/profile.d/modules.sh
module load gcc/7.2.0 || exit 1
module use /home/hltcoe/draj/modulefiles || exit 1
module load cuda || exit 1  # loads CUDA 10.2
module load cudnn || exit 1 # loads cuDNN 8.0.2
module load intel/mkl/64/2019/5.281 || exit 1
module load nccl || exit 1
