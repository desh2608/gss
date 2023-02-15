import functools
import logging
import time
from pathlib import Path

import click
from lhotse import Recording, SupervisionSet, load_manifest_lazy
from lhotse.audio import set_audio_duration_mismatch_tolerance
from lhotse.cut import CutSet
from lhotse.utils import fastcopy

from gss.bin.modes.cli_base import cli
from gss.core.enhancer import get_enhancer
from gss.utils.data_utils import post_process_manifests

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
        "--channels",
        "-c",
        type=str,
        default=None,
        help="Channels to use for enhancement. Specify with comma-separated values, e.g. "
        "`--channels 0,2,4`. All channels will be used by default.",
    )
    @click.option(
        "--select-channels-by-count",
        type=int,
        default=None,
        help="Number of channels to select for enhancement. If specified, we will use an envelope "
        "variance based method to select the best channels. If `--channels` is also specified, we will "
        "select the best channels from the specified channels.",
    )
    @click.option(
        "--select-channels-by-ratio",
        type=click.FloatRange(0.0, 1.0),
        default=None,
        help="Ratio of channels to select for enhancement. If specified, we will use an envelope "
        "variance based method to select the best channels. If `--channels` is also specified, we will "
        "select the best channels from the specified channels. Note that we will use at least 2 channels "
        "and at most the number of channels given by `--channels`.",
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
        "--use-wpe/--no-wpe",
        default=True,
        help="Whether to use WPE for GSS",
        show_default=True,
    )
    @click.option(
        "--context-duration",
        type=float,
        default=15.0,
        help="Context duration in seconds for CACGMM",
        show_default=True,
    )
    @click.option(
        "--use-garbage-class/--no-garbage-class",
        default=False,
        help="Whether to use the additional noise class for CACGMM",
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
    @click.option(
        "--max-batch-duration",
        type=float,
        default=20.0,
        help="Maximum duration of a batch in seconds",
        show_default=True,
    )
    @click.option(
        "--max-batch-cuts",
        type=int,
        default=None,
        help="Maximum number of cuts in a batch",
        show_default=True,
    )
    @click.option(
        "--num-workers",
        type=int,
        default=1,
        help="Number of workers for parallel processing",
        show_default=True,
    )
    @click.option(
        "--num-buckets",
        type=int,
        default=2,
        help="Number of buckets per speaker for batching (use larger values if you set higer max-segment-length)",
        show_default=True,
    )
    @click.option(
        "--enhanced-manifest",
        "-o",
        type=click.Path(),
        default=None,
        help="Path to the output manifest containing details of the enhanced segments.",
    )
    @click.option(
        "--profiler-output",
        type=click.Path(),
        default=None,
        help="Path to the profiler output file.",
    )
    @click.option(
        "--force-overwrite",
        is_flag=True,
        default=False,
        help="If set, we will overwrite the enhanced audio files if they already exist.",
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
@click.option(
    "--duration-tolerance",
    type=float,
    default=None,
    help="Maximum mismatch between channel durations to allow. Some corpora like CHiME-6 "
    "need a large value, e.g., 2 seconds",
)
def cuts_(
    cuts_per_recording,
    cuts_per_segment,
    enhanced_dir,
    channels,
    select_channels_by_count,
    select_channels_by_ratio,
    bss_iterations,
    use_wpe,
    context_duration,
    use_garbage_class,
    min_segment_length,
    max_segment_length,
    max_batch_duration,
    max_batch_cuts,
    num_workers,
    num_buckets,
    enhanced_manifest,
    profiler_output,
    force_overwrite,
    duration_tolerance,
):
    """
    Enhance segments (represented by cuts).

        CUTS_PER_RECORDING: Lhotse cuts manifest containing cuts per recording
        CUTS_PER_SEGMENT: Lhotse cuts manifest containing cuts per segment (e.g. obtained using `trim-to-supervisions`)
        ENHANCED_DIR: Output directory for enhanced audio files
    """
    if profiler_output is not None:
        import atexit
        import cProfile
        import pstats

        print("Profiling...")
        pr = cProfile.Profile()
        pr.enable()

        def exit():
            pr.disable()
            print("Profiling completed")
            pstats.Stats(pr).sort_stats("cumulative").dump_stats(profiler_output)

        atexit.register(exit)

    assert not (
        select_channels_by_count and select_channels_by_ratio
    ), "Please specify at most one of --select-channels-by-count and --select-channels-by-ratio"

    if duration_tolerance is not None:
        set_audio_duration_mismatch_tolerance(duration_tolerance)

    enhanced_dir = Path(enhanced_dir)
    enhanced_dir.mkdir(exist_ok=True, parents=True)

    cuts = load_manifest_lazy(cuts_per_recording)
    cuts_per_segment = load_manifest_lazy(cuts_per_segment)

    if channels is not None:
        channels = [int(c) for c in channels.split(",")]
        cuts_per_segment = CutSet.from_cuts(
            fastcopy(cut, channel=channels) for cut in cuts_per_segment
        )

    # Paranoia mode: ensure that cuts_per_recording have ids same as the recording_id
    cuts = CutSet.from_cuts(cut.with_id(cut.recording_id) for cut in cuts)

    logger.info("Aplying min/max segment length constraints")
    cuts_per_segment = cuts_per_segment.filter(
        lambda c: c.duration > min_segment_length
    ).cut_into_windows(duration=max_segment_length)

    logger.info("Initializing GSS enhancer")
    enhancer = get_enhancer(
        cuts=cuts,
        bss_iterations=bss_iterations,
        context_duration=context_duration,
        activity_garbage_class=use_garbage_class,
        wpe=use_wpe,
    )

    logger.info(f"Enhancing {len(frozenset(c.id for c in cuts_per_segment))} segments")
    begin = time.time()
    num_errors, out_cuts = enhancer.enhance_cuts(
        cuts_per_segment,
        enhanced_dir,
        max_batch_duration=max_batch_duration,
        max_batch_cuts=max_batch_cuts,
        num_workers=num_workers,
        num_buckets=num_buckets,
        force_overwrite=force_overwrite,
        select_channels_by_count=select_channels_by_count,
        select_channels_by_ratio=select_channels_by_ratio,
    )
    end = time.time()
    logger.info(f"Finished in {end-begin:.2f}s with {num_errors} errors")

    if enhanced_manifest is not None:
        logger.info(f"Saving enhanced cuts manifest to {enhanced_manifest}")
        out_cuts = post_process_manifests(out_cuts, enhanced_dir)
        out_cuts.to_file(enhanced_manifest)


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
@click.option(
    "--recording-id",
    type=str,
    default=None,
    help="Name of recording (will be used to get corresponding segments from RTTM)",
)
@common_options
def recording_(
    recording,
    rttm,
    enhanced_dir,
    recording_id,
    channels,
    select_channels_by_count,
    select_channels_by_ratio,
    bss_iterations,
    use_wpe,
    context_duration,
    use_garbage_class,
    min_segment_length,
    max_segment_length,
    max_batch_duration,
    max_batch_cuts,
    num_workers,
    num_buckets,
    enhanced_manifest,
    profiler_output,
    force_overwrite,
):
    """
    Enhance a single recording using an RTTM file.

        RECORDING: Path to a multi-channel recording
        RTTM: Path to an RTTM file containing speech activity
        ENHANCED_DIR: Output directory for enhanced audio files
    """
    if profiler_output is not None:
        import atexit
        import cProfile
        import pstats

        print("Profiling...")
        pr = cProfile.Profile()
        pr.enable()

        def exit():
            pr.disable()
            print("Profiling completed")
            pstats.Stats(pr).sort_stats("cumulative").dump_stats(profiler_output)

        atexit.register(exit)

    assert not (
        select_channels_by_count and select_channels_by_ratio
    ), "Please specify at most one of --select-channels-by-count and --select-channels-by-ratio"

    enhanced_dir = Path(enhanced_dir)
    enhanced_dir.mkdir(exist_ok=True, parents=True)

    cut = Recording.from_file(recording, recording_id=recording_id).to_cut()
    if channels is not None:
        channels = [int(c) for c in channels.split(",")]
        cut = fastcopy(cut, channel=channels)

    supervisions = SupervisionSet.from_rttm(rttm).filter(
        lambda s: s.recording_id == cut.id
    )
    # Modify channel IDs to match the recording
    supervisions = SupervisionSet.from_segments(
        fastcopy(s, channel=cut.channel) for s in supervisions
    )
    cut.supervisions = supervisions

    # Create a cuts manifest with a single cut for the recording
    cuts = CutSet.from_cuts([cut])

    # Create segment-wise cuts
    cuts_per_segment = cuts.trim_to_supervisions(
        keep_overlapping=False, keep_all_channels=True
    )

    logger.info("Aplying min/max segment length constraints")
    cuts_per_segment = cuts_per_segment.filter(
        lambda c: c.duration > min_segment_length
    ).cut_into_windows(duration=max_segment_length)

    logger.info("Initializing GSS enhancer")
    enhancer = get_enhancer(
        cuts=cuts,
        bss_iterations=bss_iterations,
        context_duration=context_duration,
        activity_garbage_class=use_garbage_class,
        wpe=use_wpe,
    )

    logger.info(f"Enhancing {len(frozenset(c.id for c in cuts_per_segment))} segments")
    begin = time.time()
    num_errors, out_cuts = enhancer.enhance_cuts(
        cuts_per_segment,
        enhanced_dir,
        max_batch_duration=max_batch_duration,
        max_batch_cuts=max_batch_cuts,
        num_workers=num_workers,
        num_buckets=num_buckets,
        force_overwrite=force_overwrite,
        select_channels_by_count=select_channels_by_count,
        select_channels_by_ratio=select_channels_by_ratio,
    )
    end = time.time()
    logger.info(f"Finished in {end-begin:.2f}s with {num_errors} errors")

    if enhanced_manifest is not None:
        logger.info(f"Saving enhanced cuts manifest to {enhanced_manifest}")
        out_cuts = post_process_manifests(out_cuts, enhanced_dir)
        out_cuts.to_file(enhanced_manifest)
