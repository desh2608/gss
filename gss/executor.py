import logging

import plz

from gss.core.enhancer import get_enhancer

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.DEBUG,
)

# The parallel execution code here is modified from:
# https://github.com/lhotse-speech/lhotse/blob/825a88484a130fa48f9a10ab214c725912b1d520/lhotse/cut.py#L3950
def run_enhancer(
    cuts,
    exp_dir,
    num_jobs=1,
    error_handling="ignore",
):
    """
    Wrapper for the enhancement.
    Args:
        cuts: a CutSet object
        exp_dir: the experiment directory (to store the enhanced cuts)
        num_jobs: number of jobs to run in parallel
        executor: an executor (for example, a Dask distributed Client)
        error_handling: how to handle errors (one of "ignore" or "keep_original")
        progress_bar: whether to show a progress bar
    """
    enhancer = get_enhancer(cuts=cuts, error_handling=error_handling)

    cuts = cuts.trim_to_supervisions(keep_overlapping=False)

    # Parallel execution: prepare the CutSet splits
    cut_sets = cuts.split(num_jobs, shuffle=False)

    # Submit the chunked tasks to parallel workers.
    # Each worker runs the non-parallel version of this function inside.
    num_errors = plz.map(
        enhancer.enhance_cuts,
        cut_sets,
        [exp_dir / "enhanced" for _ in range(num_jobs)],
        log_dir=exp_dir / "logs",
        memory="4G",
        jobs=num_jobs,
    )

    # Wait for all the tasks to finish
    num_errors = sum(num_errors)

    logging.info(f"Finished enhancing {len(cuts)} cuts. {num_errors} cuts had errors.")
    return
