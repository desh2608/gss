#!/bin/bash
# This script is used to run the enhancement.
set -euo pipefail
nj=8
stage=0

. ./path.sh
. parse_options.sh

CORPUS_DIR=/export/corpora5/amicorpus
DATA_DIR=data/
EXP_DIR=exp/ami_train

cmd="queue-ackgpu.pl --gpu 1 --mem 4G --config conf/gpu.conf"

if [ $stage -le 0 ]; then
    echo "Stage 0: Prepare manifests"
    lhotse prepare ami --mic mdm --partition full-corpus-asr $CORPUS_DIR $DATA_DIR
fi

if [ $stage -le 1 ]; then
    echo "Stage 1: Prepare cut set"
    lhotse cut simple \
      -r $DATA_DIR/ami-mdm_recordings_train.jsonl.gz \
      -s $DATA_DIR/ami-mdm_supervisions_train.jsonl.gz \
      $EXP_DIR/cuts.jsonl.gz
fi

if [ $stage -le 2 ]; then
    echo "Stage 2: Trim cuts to supervisions (1 cut per supervision segment)"
    lhotse cut trim-to-supervisions --discard-overlapping \
        $EXP_DIR/cuts.jsonl.gz $EXP_DIR/cuts_per_segment.jsonl.gz
fi

if [ $stage -le 3 ]; then
    echo "Stage 3: Split segments into $nj parts"
    lhotse split --no-pad $nj $EXP_DIR/cuts_per_segment.jsonl.gz $EXP_DIR/split$nj
fi

if [ $stage -le 4 ]; then
    echo "Stage 4: Enhance segments using GSS"
    $cmd JOB=1:$nj $EXP_DIR/log/enhance.JOB.log \
        gss enhance cuts \
          $EXP_DIR/cuts.jsonl.gz $EXP_DIR/split$nj/cuts_per_segment.JOB.jsonl.gz \
          $EXP_DIR/enhanced \
          --num-channels 7 \
          --bss-iterations 10 \
          --min-segment-length 0.0 \
          --max-segment-length 15.0
fi
