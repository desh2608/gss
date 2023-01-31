#!/bin/bash
# This script is used to run the enhancement.
set -euo pipefail
nj=4
affix=""
stage=0
stop_stage=100

. ./path.sh
. parse_options.sh

# Append _ to affix if not empty
affix=${affix:+_$affix}

CORPUS_DIR=/export/corpora6/DiPCo/DiPCo
DATA_DIR=data/
EXP_DIR=exp/dipco${affix}

cmd="queue-ackgpu.pl --gpu 1 --mem 8G --config conf/gpu.conf"

mkdir -p $DATA_DIR
mkdir -p $EXP_DIR/{dev,eval}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  echo "Stage 0: Prepare manifests"
  lhotse prepare dipco --mic mdm $CORPUS_DIR $DATA_DIR
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo "Stage 1: Prepare cut set"
  for part in dev eval; do
    lhotse cut simple --force-eager \
      -r $DATA_DIR/dipco-mdm_recordings_${part}.jsonl.gz \
      -s $DATA_DIR/dipco-mdm_supervisions_${part}.jsonl.gz \
      $EXP_DIR/$part/cuts.jsonl.gz
  done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo "Stage 2: Trim cuts to supervisions (1 cut per supervision segment)"
  for part in dev eval; do
    lhotse cut trim-to-supervisions --discard-overlapping \
      $EXP_DIR/$part/cuts.jsonl.gz $EXP_DIR/$part/cuts_per_segment.jsonl.gz
  done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  echo "Stage 3: Split segments into $nj parts"
  for part in dev eval; do
    gss utils split $nj $EXP_DIR/$part/cuts_per_segment.jsonl.gz $EXP_DIR/$part/split$nj
  done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  echo "Stage 4: Enhance segments using GSS (central array mics)"
  for part in dev eval; do
    $cmd JOB=1:$nj $EXP_DIR/$part/log/enhance.JOB.log \
      gss enhance cuts \
        $EXP_DIR/$part/cuts.jsonl.gz $EXP_DIR/$part/split$nj/cuts_per_segment.JOB.jsonl.gz \
        $EXP_DIR/$part/enhanced \
        --channels 0,7,14,21,28 \
        --bss-iterations 20 \
        --context-duration 15.0 \
        --use-garbage-class \
        --min-segment-length 0.0 \
        --max-segment-length 20.0 \
        --max-batch-duration 20.0 \
        --num-buckets 4 \
        --num-workers 4 \
        --force-overwrite \
        --duration-tolerance 3.0 || exit 1
  done
fi
