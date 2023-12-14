import logging
import os
import re
import subprocess
from pathlib import Path

import click

from gss.bin.modes.cli_base import cli

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@cli.group()
def utils():
    """General utilities for manipulating manifests."""
    pass


@utils.command(name="rttm-to-supervisions")
@click.argument("rttm_path", type=click.Path(exists=True))
@click.argument("out_path", type=click.Path())
@click.option(
    "--channels",
    "-c",
    type=int,
    default=1,
    help="Number of channels in the recording (supervisions will be modified to contain all these channels).",
)
def rttm_to_supervisions_(rttm_path, out_path, channels):
    """
    Convert RTTM file to Supervisions manifest.
    """
    from lhotse import SupervisionSet
    from lhotse.utils import fastcopy

    rttm_path = Path(rttm_path)
    rttm_files = rttm_path if rttm_path.is_file() else rttm_path.rglob("*.rttm")
    supervisions = SupervisionSet.from_rttm(rttm_files)
    # Supervisions obtained from RTTM files are single-channel only, so we modify the
    # ``channel`` field to share it for all channels.
    supervisions = SupervisionSet.from_segments(
        [fastcopy(s, channel=list(range(channels))) for s in supervisions]
    )
    supervisions.to_file(out_path)


@utils.command(name="gpu_check")
@click.argument("num_jobs", type=int)
@click.argument("cmd", type=str)
def gpu_check_(num_jobs, cmd):
    if cmd == "run.pl" and num_jobs > 1:
        used_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
        for device in used_devices:
            grep_res = subprocess.check_output(("nvidia-smi", "-i", f"{device}", "-q"))
            check = re.findall("Compute Mode\s+:\sDefault", str(grep_res))
            if not len(check) == 0:
                logging.error(
                    "This code may not work as expected with multiple GPUs "
                    f"and non exclusive process compute mode."
                    f" GPU {device} is in Default mode."
                    f" Please switch compute mode using nvidia-smi."
                )
                raise RuntimeError(
                    f"GPU {device} not in exclusive process compute mode."
                )


@utils.command(name="split")
@click.argument("num_splits", type=int)
@click.argument(
    "manifest", type=click.Path(exists=True, dir_okay=False, allow_dash=True)
)
@click.argument("output_dir", type=click.Path())
def split_(num_splits, manifest, output_dir):
    """
    This is similar to Lhotse's split command, but we additionally try to ensure that
    cuts from the same recording and speaker stay in the same split as much as possible.
    This is done by sorting the cuts by recording ID and speaker ID, and then splitting
    them into chunks of approximately equal size.
    """
    from lhotse import CutSet
    from lhotse.serialization import load_manifest_lazy_or_eager

    output_dir = Path(output_dir)
    manifest = Path(manifest)
    suffix = "".join(manifest.suffixes)
    cuts = load_manifest_lazy_or_eager(manifest)

    # sort cuts by recording ID and speaker ID
    cuts = CutSet.from_cuts(
        sorted(cuts, key=lambda c: (c.recording_id, c.supervisions[0].speaker))
    )
    parts = cuts.split(num_splits=num_splits, shuffle=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, part in enumerate(parts):
        part.to_file((output_dir / manifest.stem).with_suffix(f".{str(idx+1)}{suffix}"))
