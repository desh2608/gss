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
import soundfile as sf

from lhotse.utils import compute_num_samples

from gss.utils.data_utils import activity_time_to_frequency, start_end_context_frames
from gss.core import WPE, GSS, Beamformer, Activity

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
    bf="mvdrSouden_ban",
    postfilter=None,
    error_handling="ignore",
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
            type=bf,
            postfilter=postfilter,
        ),
        stft_size=stft_size,
        stft_shift=stft_shift,
        stft_fading=stft_fading,
        sampling_rate=sampling_rate,
        error_handling=error_handling,
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

    error_handling: str = "ignore"

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

    def enhance_cuts(self, cuts, exp_dir):
        """
        Enhance the given CutSet.
        """
        num_error = 0
        # Get cuts with extended context
        cuts_extended = cuts.extend(
            duration=self.context_duration, direction="both", preserve_id=True
        )
        for id in cuts.ids:
            cut, cut_extended = cuts[id], cuts_extended[id]
            out_dir = exp_dir / cut.recording_id
            out_dir.mkdir(parents=True, exist_ok=True)
            save_path = f"{cut.recording_id}-{cut.supervisions[0].speaker}-{int(100*cut.start):06d}_{int(100*cut.end):06d}.flac"
            logging.info(
                f"Enhancing cut {cut.id}: {cut.recording_id} ({cut.start}s to {cut.end}s), speaker {cut.supervisions[0].speaker}"
            )
            try:
                # Compute the speaker activity for the extended cut
                cut_activity, spk_to_idx_map = self.activity.get_activity(
                    cut.recording_id, cut_extended.start, cut_extended.duration
                )
                logging.debug(f"Computed speaker activity for cut")

                # Compute the enhanced signal using the extended cut and the activity
                x_hat = self.enhance_cut(
                    cut,
                    cut_extended,
                    cut_activity,
                    spk_to_idx_map[cut.supervisions[0].speaker],
                )
            except Exception as e:
                logging.exception(f"Error enhancing cut {cut.id}: {e}")
                num_error += 1
                if self.error_handling == "keep_original":
                    # Keep the original signal (this function will only load channel 0)
                    x_hat = cut.load_audio()
                raise

            logging.debug("Saving enhanced signal")
            sf.write(
                file=str(out_dir / save_path),
                data=x_hat.transpose(),
                samplerate=self.sampling_rate,
                format="FLAC",
            )
        return num_error

    def enhance_cut(self, cut, cut_extended, activity, speaker_id):

        # We load from the recording so that we can load all channels
        logging.debug("Loading audio")
        obs = cut_extended.recording.load_audio(
            offset=cut_extended.start, duration=cut_extended.duration
        )

        logging.debug(f"Computing STFT")
        Obs = self.stft(obs)

        logging.debug(f"Applying WPE")
        if self.wpe_block is not None:
            Obs = self.wpe_block(Obs)

        logging.debug(f"Converting activity to frequency domain")
        activity_freq = activity_time_to_frequency(
            activity,
            stft_window_length=self.stft_size,
            stft_shift=self.stft_shift,
            stft_fading=self.stft_fading,
            stft_pad=True,
        )

        logging.debug(f"Computing GSS masks")
        masks = self.gss_block(Obs, activity_freq)

        orig_start = compute_num_samples(cut.start, self.sampling_rate)
        orig_end = compute_num_samples(cut.end, self.sampling_rate)
        new_start = compute_num_samples(cut_extended.start, self.sampling_rate)
        new_end = compute_num_samples(cut_extended.end, self.sampling_rate)
        start_context = orig_start - new_start
        end_context = new_end - orig_end
        if self.bf_drop_context:
            logging.debug("Dropping context for beamforming")
            start_context_frames, end_context_frames = start_end_context_frames(
                start_context,
                end_context,
                stft_size=self.stft_size,
                stft_shift=self.stft_shift,
                stft_fading=self.stft_fading,
            )
            logging.debug(
                f"start_context_frames: {start_context_frames}, end_context_frames: {end_context_frames}"
            )

            masks[:, :start_context_frames, :] = 0
            if end_context_frames > 0:
                masks[:, -end_context_frames:, :] = 0

        target_mask = masks[speaker_id]
        distortion_mask = np.sum(
            np.delete(masks, speaker_id, axis=0),
            axis=0,
        )

        logging.debug("Applying beamforming with computed masks")
        X_hat = self.bf_block(
            Obs,
            target_mask=target_mask,
            distortion_mask=distortion_mask,
        )

        logging.debug("Computing inverse STFT")
        x_hat = self.istft(X_hat)

        if x_hat.ndim == 1:
            x_hat = x_hat[np.newaxis, :]

        # Trim x_hat to original length of cut
        x_hat = x_hat[:, start_context:-end_context]

        return x_hat
