#!/usr/local/env python
# -*- coding: utf-8 -*-
# Copyright 2022  Johns Hopkins University (Author: Desh Raj)
# Apache 2.0
#
# This script performs GSS-based enhancement on LibriCSS data. An RTTM file is expected
# as input. If not provided, an oracle RTTM file is constructed based on the annotations.
#
# Usage:
#   python run_libricss.py -j 30 -r data/libricss/rttm /export/data/LibriCSS exp/

from pathlib import Path
from itertools import groupby
import argparse
import logging

import sys, os

from lhotse import load_manifest
from lhotse.recipes import prepare_libricss
from gss.core.enhancer import run_enhancer

import plz

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def read_args():
    parser = argparse.ArgumentParser(description="LibriCSS multi-channel enhancement")
    parser.add_argument("corpus_dir", type=str, help="Path to LibriCSS corpus")
    parser.add_argument("exp_dir", type=str, help="Path to experiment directory")
    parser.add_argument(
        "--rttm-dir",
        "-r",
        type=str,
        default=None,
        help="Directory containing RTTM files",
    )
    parser.add_argument("--num-jobs", "-j", type=int, default=1, help="Number of jobs")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup Dask logs")
    args = parser.parse_args()
    return args


def main(args):
    logger.info("Preparing LibriCSS manifests")
    corpus_dir = Path(args.corpus_dir)

    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(exist_ok=True, parents=True)

    if (exp_dir / "recordings.jsonl").exists():
        manifests = {
            "recordings": load_manifest(exp_dir / "recordings.jsonl"),
            "supervisions": load_manifest(exp_dir / "supervisions.jsonl"),
        }
    else:
        manifests = prepare_libricss(corpus_dir)
        manifests["recordings"].to_jsonl(exp_dir / "recordings.jsonl")
        manifests["supervisions"].to_jsonl(exp_dir / "supervisions.jsonl")

    if args.rttm_dir:
        rttm_dir = Path(args.rttm_dir)
    elif (exp_dir / "rttm").exists():
        logger.info("Using existing RTTM files in %s", exp_dir / "rttm")
        rttm_dir = exp_dir / "rttm"
    else:
        logger.info("No rttm_dir given, preparing RTTMs from manifests")
        rttm_dir = exp_dir / "rttm"
        rttm_dir.mkdir(exist_ok=True, parents=True)
        rttm_string = "SPEAKER {recording_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>"
        supervisions = sorted(
            manifests["supervisions"], key=lambda x: (x.recording_id, x.start)
        )
        reco_to_supervision = {
            k: list(g) for k, g in groupby(supervisions, key=lambda x: x.recording_id)
        }
        for recording_id, supervisions in reco_to_supervision.items():
            with open(rttm_dir / f"{recording_id}.rttm", "w") as f:
                for supervision in supervisions:
                    start = supervision.start
                    duration = supervision.duration
                    speaker = supervision.speaker
                    f.write(rttm_string.format(**locals()))
                    f.write("\n")

    # Create iterable of (recording_id, rttm_path, out_path)
    iterable = list(
        (r, rttm_dir / f"{r.id}.rttm", exp_dir / "enhanced" / f"{r.id}")
        for r in manifests["recordings"]
    )

    plz.map(
        run_enhancer,
        iterable,
        jobs=args.num_jobs,
        memory="4G",
        log_dir=exp_dir / "logs",
    )

    # Cleanup Dask logs
    if args.cleanup:
        (exp_dir / "logs").rmdir()


if __name__ == "__main__":
    args = read_args()
    main(args)
