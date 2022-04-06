#!/bin/bash
# This script is used to run the enhancement.
set -euo pipefail
nj=8
rttm_tag="oracle"

. ./path.sh
. parse_options.sh

CORPUS_DIR=/export/c01/corpora6/LibriCSS
DATA_DIR=data/libricss
EXP_DIR=exp/libricss_${rttm_tag}

cmd="queue-ackgpu.pl --gpu 1 --mem 4G --config conf/gpu.conf"

rttm_opts=""
if [ $rttm_tag != "oracle" ]; then
    rttm_opts="-r ${DATA_DIR}/$rttm_tag"
fi

# Prepare data
python scripts/prepare_libricss.py $rttm_opts -j $nj $CORPUS_DIR $EXP_DIR

# Run enhancement
$cmd JOB=1:$nj $EXP_DIR/log/enhance.JOB.log \
    python enhance.py --cuts-all $EXP_DIR/cuts.jsonl \
    --cuts-split $EXP_DIR/split${nj}/cuts.JOB.jsonl \
    --out-dir $EXP_DIR/enhanced

