#!/bin/bash
# This script is used to run the enhancement.
set -euo pipefail
nj=8
rttm_dir=""
affix="_oracle"
stage=0
stop_stage=100

. ./path.sh
. parse_options.sh

CORPUS_DIR=/export/c01/corpora6/AliMeeting
DATA_DIR=data/
EXP_DIR=exp/alimeeting${affix}

cmd="queue-ackgpu.pl --gpu 1 --mem 4G --config conf/gpu.conf"

mkdir -p $DATA_DIR
mkdir -p $EXP_DIR

if [ -z $rttm_dir ]; then
    supervisions_path=$DATA_DIR/alimeeting_supervisions_all.jsonl.gz
else
    supervisions_path=$EXP_DIR/supervisions_${rttm_tag}.jsonl.gz
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Prepare manifests"
    lhotse prepare ali-meeting --mic far $CORPUS_DIR $DATA_DIR
    lhotse combine $DATA_DIR/alimeeting_recordings_{eval,test}.jsonl.gz data/alimeeting_recordings_all.jsonl.gz
    lhotse combine $DATA_DIR/alimeeting_supervisions_{eval,test}.jsonl.gz data/alimeeting_supervisions_all.jsonl.gz
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ] && [ ! -z $rttm_dir ]; then
    echo "Stage 1: Create supervisions from RTTM file"
    gss utils rttm-to-supervisions --channels 7 $rttm_dir $supervisions_path
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 2: Prepare cut set"
    lhotse cut simple --force-eager \
      -r $DATA_DIR/alimeeting_recordings_all.jsonl.gz \
      -s $supervisions_path \
      $EXP_DIR/cuts.jsonl.gz
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "Stage 3: Trim cuts to supervisions (1 cut per supervision segment)"
    lhotse cut trim-to-supervisions --discard-overlapping \
        $EXP_DIR/cuts.jsonl.gz $EXP_DIR/cuts_per_segment.jsonl.gz
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Stage 4: Split segments into $nj parts"
    gss utils split --no-pad $nj $EXP_DIR/cuts_per_segment.jsonl.gz $EXP_DIR/split$nj
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    echo "Stage 5: Enhance segments using GSS"
    $cmd JOB=1:$nj $EXP_DIR/log/enhance.JOB.log \
        gss enhance cuts \
          $EXP_DIR/cuts.jsonl.gz $EXP_DIR/split$nj/cuts_per_segment.JOB.jsonl.gz \
          $EXP_DIR/enhanced \
          --num-channels 7 \
          --bss-iterations 10 \
          --min-segment-length 0.0 \
          --max-segment-length 15.0 \
          --max-batch-duration 20.0 \
          --num-buckets 3 \
          --enhanced-manifest $EXP_DIR/split$nj/cuts_enhanced.JOB.jsonl.gz

    echo "Stage 5: Combine enhanced cuts"
    lhotse combine $EXP_DIR/split$nj/cuts_enhanced.*.jsonl.gz $EXP_DIR/cuts_enhanced.jsonl.gz
fi
