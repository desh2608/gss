import functools
import logging
from pathlib import Path

import click

from gss.bin.modes.cli_base import cli
from gss.core.enhancer import get_enhancer

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@cli.group()
def enhance():
    """Commands for enhancing single recordings or manifests."""
    pass


def common_options(func):
    @click.option(
        "--num-channels",
        "-c",
        type=int,
        default=None,
        help="Number of channels to use for enhancement. All channels will be used by default.",
    )
    @click.option(
        "--bss-iterations",
        "-i",
        type=int,
        default=10,
        help="Number of iterations for BSS",
        show_default=True,
    )
    @click.option(
        "--min-segment-length",
        type=float,
        default=0.0,
        help="Minimum segment length to retain (removing very small segments speeds up enhancement)",
        show_default=True,
    )
    @click.option(
        "--max-segment-length",
        type=float,
        default=15.0,
        help="Chunk up longer segments to avoid OOM issues",
        show_default=True,
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@enhance.command(name="cuts")
@click.argument(
    "cuts_per_recording",
    type=click.Path(exists=True),
)
@click.argument(
    "cuts_per_segment",
    type=click.Path(exists=True),
)
@click.argument(
    "enhanced_dir",
    type=click.Path(),
)
@common_options
def cuts_(
    cuts_per_recording,
    cuts_per_segment,
    enhanced_dir,
    num_channels,
    bss_iterations,
    min_segment_length,
    max_segment_length,
):
    """
    Enhance segments (represented by cuts).

        CUTS_PER_RECORDING: Lhotse cuts manifest containing cuts per recording
        CUTS_PER_SEGMENT: Lhotse cuts manifest containing cuts per segment (e.g. obtained using `trim-to-supervisions`)
        ENHANCED_DIR: Output directory for enhanced audio files
    """
    from lhotse import load_manifest

    enhanced_dir = Path(enhanced_dir)
    enhanced_dir.mkdir(exist_ok=True, parents=True)

    cuts = load_manifest(cuts_per_recording)
    cuts_per_segment = load_manifest(cuts_per_segment)

    logger.info("Aplying min/max segment length constraints")
    cuts_per_segment = (
        cuts_per_segment.filter(lambda c: c.duration > min_segment_length)
        .cut_into_windows(duration=max_segment_length)
        .to_eager()
    )

    logger.info("Initializing GSS enhancer")
    enhancer = get_enhancer(
        cuts=cuts,
        error_handling="keep_original",
        activity_garbage_class=False,
        bss_iterations=bss_iterations,
        num_channels=num_channels,
    )

    logger.info(f"Enhancing {len(cuts_per_segment)} segments")
    num_errors = enhancer.enhance_cuts(cuts_per_segment, enhanced_dir)
    logger.info(f"Finished with {num_errors} errors")


@enhance.command(name="recording")
@click.argument(
    "recording",
    type=click.Path(exists=True),
)
@click.argument(
    "rttm",
    type=click.Path(exists=True),
)
@click.argument(
    "enhanced_dir",
    type=click.Path(),
)
# @common_options
def recording_(
    recording,
    rttm,
    enhanced_dir,
    num_channels,
    bss_iterations,
    min_segment_length,
    max_segment_length,
):
    """
    Enhance a single recording using an RTTM file.

        RECORDING: Path to a multi-channel recording
        RTTM: Path to an RTTM file containing speech activity
        ENHANCED_DIR: Output directory for enhanced audio files
    """
    from lhotse import CutSet, Recording, RecordingSet, SupervisionSet
    from lhotse.utils import fastcopy

    enhanced_dir = Path(enhanced_dir)
    enhanced_dir.mkdir(exist_ok=True, parents=True)

    recordings = RecordingSet.from_recordings(Recording.from_file(recording))
    supervisions = SupervisionSet.from_rttm(rttm)
    # Modify channel IDs to match the recording
    supervisions = SupervisionSet.from_segments(
        fastcopy(s, channel=recording.channel_ids) for s in supervisions
    )

    # Create a cuts manifest with a single cut for the recording
    cuts = CutSet.from_manifests(recordings=recordings, supervisions=supervisions)

    # Create segment-wise cuts
    cuts_per_segment = cuts.trim_to_supervisions(keep_overlapping=False)

    logger.info("Aplying min/max segment length constraints")
    cuts_per_segment = (
        cuts_per_segment.filter(lambda c: c.duration > min_segment_length)
        .cut_into_windows(duration=max_segment_length)
        .to_eager()
    )

    logger.info("Initializing GSS enhancer")
    enhancer = get_enhancer(
        cuts=cuts,
        error_handling="keep_original",
        activity_garbage_class=False,
        bss_iterations=bss_iterations,
        num_channels=num_channels,
    )

    logger.info(f"Enhancing {len(cuts_per_segment)} segments")
    num_errors = enhancer.enhance_cuts(cuts_per_segment, enhanced_dir)
    logger.info(f"Finished with {num_errors} errors")
