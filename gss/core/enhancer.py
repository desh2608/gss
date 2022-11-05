import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cupy as cp
import numpy as np
import soundfile as sf
from lhotse import CutSet, Recording, RecordingSet, SupervisionSegment, SupervisionSet
from lhotse.utils import add_durations, compute_num_samples
from torch.utils.data import DataLoader

from gss.core import GSS, WPE, Activity, Beamformer
from gss.utils.data_utils import (
    GssDataset,
    activity_time_to_frequency,
    create_sampler,
    start_end_context_frames,
)

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def get_enhancer(
    cuts,
    context_duration=15,  # 15 seconds
    wpe=True,
    wpe_tabs=10,
    wpe_delay=2,
    wpe_iterations=3,
    wpe_psd_context=0,
    activity_garbage_class=True,
    stft_size=1024,
    stft_shift=256,
    stft_fading=True,
    bss_iterations=20,
    bss_iterations_post=1,
    bf_drop_context=True,
    postfilter=None,
):
    assert wpe is True or wpe is False, wpe
    assert len(cuts) > 0

    sampling_rate = cuts[0].recording.sampling_rate

    return Enhancer(
        context_duration=context_duration,
        wpe_block=WPE(
            taps=wpe_tabs,
            delay=wpe_delay,
            iterations=wpe_iterations,
            psd_context=wpe_psd_context,
        )
        if wpe
        else None,
        activity=Activity(
            garbage_class=activity_garbage_class,
            cuts=cuts,
        ),
        gss_block=GSS(
            iterations=bss_iterations,
            iterations_post=bss_iterations_post,
        ),
        bf_drop_context=bf_drop_context,
        bf_block=Beamformer(
            postfilter=postfilter,
        ),
        stft_size=stft_size,
        stft_shift=stft_shift,
        stft_fading=stft_fading,
        sampling_rate=sampling_rate,
    )


@dataclass
class Enhancer:
    """
    This class creates enhancement context (with speaker activity) for the sessions, and
    performs the enhancement.
    """

    wpe_block: WPE
    activity: Activity
    gss_block: GSS
    bf_block: Beamformer

    bf_drop_context: bool

    stft_size: int
    stft_shift: int
    stft_fading: bool

    context_duration: float  # e.g. 15
    sampling_rate: int

    def stft(self, x):
        from gss.core.stft_module import stft

        return stft(
            x,
            size=self.stft_size,
            shift=self.stft_shift,
            fading=self.stft_fading,
        )

    def istft(self, X):
        from gss.core.stft_module import istft

        return istft(
            X,
            size=self.stft_size,
            shift=self.stft_shift,
            fading=self.stft_fading,
        )

    def enhance_cuts(
        self,
        cuts,
        exp_dir,
        max_batch_duration=None,
        max_batch_cuts=None,
        num_buckets=2,
        num_workers=1,
        force_overwrite=False,
    ):
        """
        Enhance the given CutSet.
        """
        num_error = 0
        out_cuts = []  # list of enhanced cuts

        # Create the dataset, sampler, and data loader
        gss_dataset = GssDataset(
            context_duration=self.context_duration, activity=self.activity
        )
        gss_sampler = create_sampler(
            cuts,
            max_duration=max_batch_duration,
            max_cuts=max_batch_cuts,
            num_buckets=num_buckets,
        )
        dl = DataLoader(
            gss_dataset,
            sampler=gss_sampler,
            batch_size=None,
            num_workers=num_workers,
            persistent_workers=False,
        )

        def _save_worker(orig_cuts, x_hat, recording_id, speaker):
            out_dir = exp_dir / recording_id
            enhanced_recordings = []
            enhanced_supervisions = []
            offset = 0
            for cut in orig_cuts:
                save_path = Path(
                    f"{recording_id}-{speaker}-{int(100*cut.start):06d}_{int(100*cut.end):06d}.flac"
                )
                if force_overwrite or not (out_dir / save_path).exists():
                    st = compute_num_samples(offset, self.sampling_rate)
                    en = st + compute_num_samples(cut.duration, self.sampling_rate)
                    x_hat_cut = x_hat[:, st:en]
                    logging.debug("Saving enhanced signal")
                    sf.write(
                        file=str(out_dir / save_path),
                        data=x_hat_cut.transpose(),
                        samplerate=self.sampling_rate,
                        format="FLAC",
                    )
                    # Update offset for the next cut
                    offset = add_durations(
                        offset, cut.duration, sampling_rate=self.sampling_rate
                    )
                else:
                    logging.info(f"File {save_path} already exists. Skipping.")
                # add enhanced recording to list
                enhanced_recordings.append(Recording.from_file(out_dir / save_path))
                # modify supervision channels since enhanced recording has only 1 channel
                enhanced_supervisions.extend(
                    [
                        SupervisionSegment(
                            id=str(save_path),
                            recording_id=str(save_path),
                            start=segment.start,
                            duration=segment.duration,
                            channel=0,
                            text=segment.text,
                            language=segment.language,
                            speaker=segment.speaker,
                        )
                        for segment in cut.supervisions
                    ]
                )
            return enhanced_recordings, enhanced_supervisions

        # Iterate over batches
        futures = []
        total_processed = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for batch_idx, batch in enumerate(dl):
                batch = SimpleNamespace(**batch)
                logging.info(
                    f"Processing batch {batch_idx+1} {batch.recording_id, batch.speaker}: "
                    f"{len(batch.orig_cuts)} segments = {batch.duration}s (total: {total_processed} segments)"
                )
                total_processed += len(batch.orig_cuts)

                out_dir = exp_dir / batch.recording_id
                out_dir.mkdir(parents=True, exist_ok=True)

                file_exists = []
                if not force_overwrite:
                    for cut in batch.orig_cuts:
                        save_path = Path(
                            f"{batch.recording_id}-{batch.speaker}-{int(100*cut.start):06d}_{int(100*cut.end):06d}.flac"
                        )
                        file_exists.append((out_dir / save_path).exists())

                    if all(file_exists):
                        logging.info("All files already exist. Skipping.")
                        continue

                # Sometimes the segment may be large and cause OOM issues in CuPy. If this
                # happens, we increasingly chunk it up into smaller segments until it can
                # be processed without breaking.
                num_chunks = 1
                while True:
                    try:
                        x_hat = self.enhance_batch(
                            batch.audio,
                            batch.activity,
                            batch.speaker_idx,
                            num_chunks=num_chunks,
                            left_context=batch.left_context,
                            right_context=batch.right_context,
                        )
                        break
                    except cp.cuda.memory.OutOfMemoryError:
                        num_chunks = num_chunks + 1
                        logging.warning(
                            f"Out of memory error while processing the batch. Trying again with {num_chunks} chunks."
                        )
                    except Exception as e:
                        logging.error(f"Error enhancing batch: {e}")
                        num_error += 1
                        # Keep the original signal (only load channel 0)
                        # NOTE (@desh2608): One possible issue here is that the whole batch
                        # may fail even if the issue is only due to one segment. We may
                        # want to handle this case separately.
                        x_hat = batch.audio[0:1].cpu().numpy()
                        break

                # Save the enhanced cut to disk
                futures.append(
                    executor.submit(
                        _save_worker,
                        batch.orig_cuts,
                        x_hat,
                        batch.recording_id,
                        batch.speaker,
                    )
                )

        out_recordings = []
        out_supervisions = []
        for future in futures:
            enhanced_recordings, enhanced_supervisions = future.result()
            out_recordings.extend(enhanced_recordings)
            out_supervisions.extend(enhanced_supervisions)

        out_recordings = RecordingSet.from_recordings(out_recordings)
        out_supervisions = SupervisionSet.from_segments(out_supervisions)
        return num_error, CutSet.from_manifests(
            recordings=out_recordings, supervisions=out_supervisions
        )

    def enhance_batch(
        self, obs, activity, speaker_id, num_chunks=1, left_context=0, right_context=0
    ):

        logging.debug(f"Converting activity to frequency domain")
        activity_freq = activity_time_to_frequency(
            activity,
            stft_window_length=self.stft_size,
            stft_shift=self.stft_shift,
            stft_fading=self.stft_fading,
            stft_pad=True,
        )

        # Convert to cupy array (putting it on the GPU)
        obs = cp.asarray(obs)

        logging.debug(f"Computing STFT")
        Obs = self.stft(obs)

        D, T, F = Obs.shape

        # Process observation in chunks
        chunk_size = int(np.ceil(T / num_chunks))
        masks = []
        for i in range(num_chunks):
            st = i * chunk_size
            en = min(T, (i + 1) * chunk_size)
            Obs_chunk = Obs[:, st:en, :]

            logging.debug(f"Applying WPE")
            if self.wpe_block is not None:
                Obs_chunk = self.wpe_block(Obs_chunk)

            logging.debug(f"Computing GSS masks")
            masks_chunk = self.gss_block(Obs_chunk, activity_freq[:, st:en])
            masks.append(masks_chunk)

        masks = cp.concatenate(masks, axis=1)
        if self.bf_drop_context:
            logging.debug("Dropping context for beamforming")
            left_context_frames, right_context_frames = start_end_context_frames(
                left_context,
                right_context,
                stft_size=self.stft_size,
                stft_shift=self.stft_shift,
                stft_fading=self.stft_fading,
            )
            logging.debug(
                f"left_context_frames: {left_context_frames}, right_context_frames: {right_context_frames}"
            )

            masks[:, :left_context_frames, :] = 0
            if right_context_frames > 0:
                masks[:, -right_context_frames:, :] = 0

        target_mask = masks[speaker_id]
        distortion_mask = cp.sum(masks, axis=0) - target_mask

        logging.debug("Applying beamforming with computed masks")
        X_hat = []
        for i in range(num_chunks):
            st = i * chunk_size
            en = min(T, (i + 1) * chunk_size)
            X_hat_chunk = self.bf_block(
                Obs[:, st:en, :],
                target_mask=target_mask[st:en],
                distortion_mask=distortion_mask[st:en],
            )
            X_hat.append(X_hat_chunk)

        X_hat = cp.concatenate(X_hat, axis=0)

        logging.debug("Computing inverse STFT")
        x_hat = self.istft(X_hat)  # returns a numpy array

        if x_hat.ndim == 1:
            x_hat = x_hat[np.newaxis, :]

        # Trim x_hat to original length of cut
        x_hat = x_hat[:, left_context:-right_context]

        return x_hat
