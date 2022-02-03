"""

Legend:
n, N ... time
t, T ... frame
f, F ... frequency
d, D ... channel
a, A ... array
"""
from dataclasses import dataclass
import logging

import numpy as np
import torch
import torchaudio

from gss.utils.data_utils import activity_time_to_frequency, start_end_context_frames
from gss.dataset import RTTMDataset
from gss.core import WPE, GSS, Beamformer, Activity, activity

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.DEBUG,
)


def run_enhancer(args):
    """
    Wrapper for the enhancement. Takes as input a tuple containing (recording, RTTM, out_dir)
    and performs enhancement on them.
    """
    recording, rttm, out_dir = args
    out_dir.mkdir(parents=True, exist_ok=True)
    enhancer = get_enhancer(rttm=rttm)
    dataset = RTTMDataset(recording, rttm)
    enhancer.enhance_session(dataset, out_dir)
    # free up memory
    del enhancer
    del dataset


def get_enhancer(
    rttm,
    context_samples=240000,  # 15 seconds
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
    bf="mvdrSouden_ban",
    postfilter=None,
):
    assert wpe is True or wpe is False, wpe

    return Enhancer(
        context_samples=context_samples,
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
            rttm=rttm,
        ),
        gss_block=GSS(
            iterations=bss_iterations,
            iterations_post=bss_iterations_post,
        ),
        bf_drop_context=bf_drop_context,
        bf_block=Beamformer(
            type=bf,
            postfilter=postfilter,
        ),
        stft_size=stft_size,
        stft_shift=stft_shift,
        stft_fading=stft_fading,
    )


@dataclass
class Enhancer:
    """
    This class handles enhancement for a single session.
    """

    wpe_block: WPE
    activity: Activity
    gss_block: GSS
    bf_block: Beamformer

    bf_drop_context: bool

    stft_size: int
    stft_shift: int
    stft_fading: bool

    # context_samples: int
    # equal_start_context: bool

    context_samples: int  # e.g. 240000

    def stft(self, x):
        from paderbox.transform.module_stft import stft

        return stft(
            x,
            size=self.stft_size,
            shift=self.stft_shift,
            fading=self.stft_fading,
        )

    def istft(self, X):
        from paderbox.transform.module_stft import istft

        return istft(
            X,
            size=self.stft_size,
            shift=self.stft_shift,
            fading=self.stft_fading,
        )

    def enhance_session(self, dataset, out_dir):
        """
        Args:
            dataset: list of segments created using the `get_dataset` method
            out_dir: Path to output directory for enhanced audio files
        Returns:
        """
        num_total = 0
        num_errors = 0
        for ex in dataset.get_examples(
            audio_read=True, context_samples=self.context_samples
        ):
            num_total += 1
            try:
                example_id = ex["example_id"]
                logging.info(f"Enhancing example {example_id}")
                x_hat = self.enhance_example(ex)

                save_path = out_dir / f"{example_id}.wav"
                if x_hat.ndim == 1:
                    x_hat = x_hat[np.newaxis, :]  # add channel dimension
                torchaudio.save(
                    save_path, torch.tensor(x_hat, dtype=torch.float32), 16000
                )
            except Exception:
                logging.error(f"Failed example: {ex['example_id']}")
                num_errors += 1
                continue
        logging.info(f"Finished enhancing {num_total} examples. {num_errors} failed.")

    def enhance_example(self, ex):
        session_id = ex["session_id"]
        speaker_id = ex["speaker_id"]

        array_start = ex["start"]
        array_end = ex["end"]

        ex_array_activity = {
            k: arr[array_start:array_end]
            for k, arr in self.activity[session_id].items()
        }

        obs = ex["audio_data"]

        x_hat = self.enhance_observation(
            obs, ex_array_activity=ex_array_activity, speaker_id=speaker_id, ex=ex
        )

        if self.context_samples > 0:
            start_orig = ex["start_orig"]
            start = ex["start"]
            start_context = start_orig - start
            num_samples_orig = ex["num_samples_orig"]
            x_hat = x_hat[..., start_context : start_context + num_samples_orig]

        return x_hat

    def enhance_observation(self, obs, ex_array_activity, speaker_id, ex=None):

        # Compute STFT for observation
        Obs = self.stft(obs)

        # Apply WPE for dereverberation
        if self.wpe_block is not None:
            Obs = self.wpe_block(Obs)

        # Convert activity to frequency domain
        activity_freq = activity_time_to_frequency(
            np.array(list(ex_array_activity.values())),
            stft_window_length=self.stft_size,
            stft_shift=self.stft_shift,
            stft_fading=self.stft_fading,
            stft_pad=True,
        )

        # Apply GSS
        masks = self.gss_block(Obs, activity_freq)

        if self.bf_drop_context:
            start_context_frames, end_context_frames = start_end_context_frames(
                ex,
                stft_size=self.stft_size,
                stft_shift=self.stft_shift,
                stft_fading=self.stft_fading,
            )

            masks[:, :start_context_frames, :] = 0
            if end_context_frames > 0:
                masks[:, -end_context_frames:, :] = 0

        target_speaker_index = tuple(ex_array_activity.keys()).index(speaker_id)
        target_mask = masks[target_speaker_index]
        distortion_mask = np.sum(
            np.delete(masks, target_speaker_index, axis=0),
            axis=0,
        )

        # Apply beamforming
        X_hat = self.bf_block(
            Obs,
            target_mask=target_mask,
            distortion_mask=distortion_mask,
        )

        # Compute inverse STFT
        x_hat = self.istft(X_hat)

        return x_hat
