#!/bin/bash
# This script is used to run the enhancement.
set -euo pipefail
nj=10

. ./path.sh
. parse_options.sh

cmd="queue-freegpu.pl --gpu 1 --mem 4G"

# LibriCSS

# Prepare data
# python scripts/prepare_libricss.py -j $nj /export/c01/corpora6/LibriCSS exp/libricss_oracle_v2

# Run enhancement
$cmd JOB=1:$nj exp/libricss_oracle_v2/log/enhance.JOB.log \
    python enhance.py --cuts-all exp/libricss_oracle_v2/cuts.jsonl \
    --cuts-split exp/libricss_oracle_v2/split${nj}/cuts.JOB.jsonl \
    --out-dir exp/libricss_oracle_v2/enhanced
