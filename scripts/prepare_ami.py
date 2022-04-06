#!/usr/local/env python
# -*- coding: utf-8 -*-
# Copyright 2022  Johns Hopkins University (Author: Desh Raj)
# Apache 2.0
#
# This script prepares cut manifests for AMI data. An RTTM file is expected
# as input. If not provided, an oracle RTTM file is constructed based on the annotations.
#
# Usage:
#   python scripts/prepare_ami.py -j 8 -r data/ami/rttm /export/data/amicorpus exp/

from pathlib import Path
import argparse
import logging

from lhotse import (
    SupervisionSet,
    CutSet,
    load_manifest,
    validate_recordings_and_supervisions,
    fix_manifests,
)
from lhotse.recipes import prepare_ami
from lhotse.utils import fastcopy

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def read_args():
    parser = argparse.ArgumentParser(description="AMI multi-channel enhancement")
    parser.add_argument("corpus_dir", type=Path, help="Path to AMI corpus")
    parser.add_argument("exp_dir", type=Path, help="Path to experiment directory")
    parser.add_argument(
        "--rttm-path",
        "-r",
        type=Path,
        default=None,
        help="Path to RTTM file (or directory containing RTTM files)",
    )
    parser.add_argument("--num-jobs", "-j", type=int, default=1, help="Number of jobs")
    parser.add_argument(
        "--min-segment-length",
        type=float,
        default=0.0,
        help="Minimum segment length to retain (removing very small segments speeds up enhancement)",
    )
    parser.add_argument(
        "--max-segment-length",
        type=float,
        default=15.0,
        help="Chunk up longer segments to avoid OOM issues",
    )
    args = parser.parse_args()
    return args


def main(args):
    logger.info("Preparing AMI manifests")
    corpus_dir = args.corpus_dir
    exp_dir = args.exp_dir
    exp_dir.mkdir(exist_ok=True, parents=True)

    if (exp_dir / "cuts.jsonl").exists():
        logger.info("Loading existing CutSet")
        cuts = load_manifest(exp_dir / "cuts.jsonl")

    else:
        manifests = prepare_ami(
            corpus_dir,
            annotations_dir=corpus_dir / "ami_public_manual_1.6.2",
            mic="mdm",
            partition="full-corpus-asr",
        )

        recordings = manifests["dev"]["recordings"] + manifests["test"]["recordings"]

        if args.rttm_path:
            logger.info("Creating supervisions from RTTM file(s)")
            rttm_path = Path(args.rttm_path)
            rttm_files = rttm_path if rttm_path.is_file() else rttm_path.rglob("*.rttm")
            supervisions = SupervisionSet.from_rttm(rttm_files)

        else:
            supervisions = (
                manifests["dev"]["supervisions"] + manifests["test"]["supervisions"]
            )

        channels = set(s.channel for s in supervisions)
        assert len(channels) == 1, "Only one channel is supported"
        channel = channels.pop()

        # Remove very short segments
        supervisions = supervisions.filter(
            lambda s: s.duration >= args.min_segment_length
        )

        recordings, supervisions = fix_manifests(recordings, supervisions)
        validate_recordings_and_supervisions(recordings, supervisions)

        logger.info("Creating CutSet")
        cuts = CutSet.from_manifests(recordings=recordings, supervisions=supervisions)
        # Only keep the cuts with channel id 0, since we only have supervisions for those
        cuts = cuts.filter(lambda c: c.channel == channel)
        # Now we change the cut ids to be the same as the corresponding recording id
        cuts = CutSet.from_cuts(fastcopy(c, id=c.recording_id) for c in cuts)
        # At this point, there is 1 cut per recording.
        cuts.to_file(exp_dir / "cuts.jsonl")

    # Create cuts corresponding to the segments provided. The enhancement will produce
    # 1 output audio file per segment.
    logger.info("Creating segments")
    cuts = cuts.trim_to_supervisions(keep_overlapping=False).cut_into_windows(
        duration=args.max_segment_length
    )

    logger.info(f"Splitting cuts into {args.num_jobs} parts")
    cut_sets = cuts.split(args.num_jobs, shuffle=False)

    logger.info("Writing cuts to disk")
    split_dir = exp_dir / f"split{args.num_jobs}"
    split_dir.mkdir(exist_ok=True, parents=True)
    for i, cut_set in enumerate(cut_sets):
        cut_set.to_file(split_dir / f"cuts.{i+1}.jsonl")


if __name__ == "__main__":
    args = read_args()
    main(args)
