#!/usr/local/env python
# -*- coding: utf-8 -*-
# Copyright 2022  Johns Hopkins University (Author: Desh Raj)
# Apache 2.0
#
# This script performs GSS-based enhancement on a prepared cutset.

from pathlib import Path
import argparse
import logging

from lhotse import load_manifest
from gss.core.enhancer import get_enhancer

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def read_args():
    parser = argparse.ArgumentParser(description="GSS multi-channel enhancement")
    parser.add_argument(
        "--cuts-all",
        type=str,
        help="Path to cuts containing all segments (used to initialize speaker activities)",
    )
    parser.add_argument(
        "--cuts-split", type=str, help="Path to cuts jsonl file to be enhanced"
    )
    parser.add_argument(
        "--out-dir", type=str, help="Path to write enhanced audio files"
    )
    parser.add_argument(
        "--num-channels",
        type=int,
        default=None,
        help="Number of channels to use (use all by default)",
    )
    args = parser.parse_args()
    return args


def main(args):
    logger.info("Loading cuts manifest")
    cuts_all = load_manifest(args.cuts_all)
    cuts_split = load_manifest(args.cuts_split)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    logger.info("Initializing GSS enhancer")
    enhancer = get_enhancer(
        cuts=cuts_all,
        error_handling="keep_original",
        activity_garbage_class=False,
        bss_iterations=10,
        num_channels=args.num_channels,
    )

    logger.info(f"Enhancing {len(cuts_split)} segments")
    num_errors = enhancer.enhance_cuts(cuts_split, out_dir)
    logger.info(f"Finished with {num_errors} errors")


if __name__ == "__main__":
    args = read_args()
    main(args)
