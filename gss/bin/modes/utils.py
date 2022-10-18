from pathlib import Path

import click

from gss.bin.modes.cli_base import cli


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
