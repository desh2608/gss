from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
from lhotse import CutSet, validate
from lhotse.cut import Cut
from lhotse.dataset.sampling.dynamic import DynamicCutSampler
from lhotse.dataset.sampling.dynamic_bucketing import DynamicBucketingSampler
from lhotse.dataset.sampling.round_robin import RoundRobinSampler
from lhotse.utils import add_durations, compute_num_samples
from torch.utils.data import Dataset

from gss.utils.numpy_utils import segment_axis


class GssDataset(Dataset):
    """
    It takes a batch of cuts as input (all from the same recording and speaker) and
    concatenates them into a single sequence. Additionally, we also extend the left
    and right cuts by the context duration, so that the model can see the context
    and disambiguate the target speaker from background noise.
    Returns:
    .. code-block::
        {
            'audio': (channels x total #samples) float tensor
            'activity': (#speakers x total #samples) int tensor denoting speaker activities
            'cuts': original cuts (sorted by start time)
            'speaker': str, speaker ID
            'recording': str, recording ID
            'start': float tensor, start times of the cuts w.r.t. concatenated sequence
        }
    In the returned tensor, the ``audio`` and ``activity`` will be used to perform the
    actual enhancement. The ``speaker``, ``recording``, and ``start`` are
    used to name the enhanced files.
    """

    def __init__(
        self, activity, context_duration: float = 0, num_channels: int = None
    ) -> None:
        super().__init__()
        self.activity = activity
        self.context_duration = context_duration
        self.num_channels = num_channels

    def __getitem__(self, cuts: CutSet) -> Dict[str, Any]:
        self._validate(cuts)

        recording_id = cuts[0].recording_id
        speaker = cuts[0].supervisions[0].speaker

        # sort cuts by start time
        orig_cuts = sorted(cuts, key=lambda cut: cut.start)

        new_cuts = orig_cuts[:]

        # Extend the first and last cuts by the context duration.
        new_cuts[0] = new_cuts[0].extend_by(
            duration=self.context_duration,
            direction="left",
            preserve_id=True,
            pad_silence=False,
        )
        left_context = orig_cuts[0].start - new_cuts[0].start
        new_cuts[-1] = new_cuts[-1].extend_by(
            duration=self.context_duration,
            direction="right",
            preserve_id=True,
            pad_silence=False,
        )
        right_context = new_cuts[-1].end - orig_cuts[-1].end

        # Concatenate the cuts into a single sequence. For each cut, we also need to store
        # its start/end time w.r.t. the concatenated sequence, so that we can
        # later split the enhanced audio into individual cuts.
        concatenated = None
        activity = []
        start_times = []
        for orig_cut, new_cut in zip(orig_cuts, new_cuts):
            concatenated = (
                new_cut
                if concatenated is None
                else concatenated.append(new_cut, preserve_id="left")
            )
            cut_activity, spk_to_idx_map = self.activity.get_activity(
                new_cut.recording_id, new_cut.start, new_cut.duration
            )
            activity.append(cut_activity)
            start_times.append(orig_cut.start - concatenated.start)

        # Load audio
        audio = concatenated.load_audio()
        activity = np.concatenate(activity, axis=1)

        return {
            "audio": audio,
            "duration": add_durations(
                *[c.duration for c in orig_cuts],
                sampling_rate=concatenated.sampling_rate
            ),
            "left_context": compute_num_samples(
                left_context, sampling_rate=concatenated.sampling_rate
            ),
            "right_context": compute_num_samples(
                right_context, sampling_rate=concatenated.sampling_rate
            ),
            "activity": activity,
            "orig_cuts": orig_cuts,
            "speaker": speaker,
            "speaker_idx": spk_to_idx_map[speaker],
            "recording_id": recording_id,
            "start_times": start_times,
        }

    def _validate(self, cuts: CutSet) -> None:
        validate(cuts)
        assert all(cut.has_recording for cut in cuts)
        assert len(cuts) > 0

        # check that all cuts have the same speaker and recording
        speaker = cuts[0].supervisions[0].speaker
        recording = cuts[0].recording_id
        assert all(cut.supervisions[0].speaker == speaker for cut in cuts)
        assert all(cut.recording_id == recording for cut in cuts)


def create_sampler(
    cuts: CutSet, max_duration: float = None, max_cuts: int = None, num_buckets: int = 1
) -> RoundRobinSampler:
    buckets = create_buckets_by_speaker(cuts)
    sampler = RoundRobinSampler(
        *[
            DynamicBucketingSampler(
                bucket,
                max_duration=max_duration,
                max_cuts=max_cuts,
                num_buckets=num_buckets,
            )
            for bucket in buckets
        ]
    )
    return sampler


def create_buckets_by_speaker(cuts: CutSet) -> List[CutSet]:
    """
    Helper method to partition a single CutSet into buckets that have the same
    recording and speaker.
    """
    buckets: Dict[Tuple[str, str], List[Cut]] = defaultdict(list)
    for cut in cuts:
        buckets[(cut.recording_id, cut.supervisions[0].speaker)].append(cut)
    return [CutSet.from_cuts(cuts) for cuts in buckets.values()]


def start_end_context_frames(
    start_context_samples, end_context_samples, stft_size, stft_shift, stft_fading
):
    assert start_context_samples >= 0
    assert end_context_samples >= 0

    from nara_wpe.utils import _samples_to_stft_frames

    start_context_frames = _samples_to_stft_frames(
        start_context_samples,
        size=stft_size,
        shift=stft_shift,
        fading=stft_fading,
    )
    end_context_frames = _samples_to_stft_frames(
        end_context_samples,
        size=stft_size,
        shift=stft_shift,
        fading=stft_fading,
    )
    return start_context_frames, end_context_frames


def activity_time_to_frequency(
    time_activity,
    stft_window_length,
    stft_shift,
    stft_fading,
    stft_pad=True,
):
    """
    >>> from nara_wpe.utils import stft
    >>> signal = np.array([0, 0, 0, 0, 0, 1, -3, 0, 5, 0, 0, 0, 0, 0])
    >>> vad = np.array(   [0, 0, 0, 0, 0, 1,  1, 0, 1, 0, 0, 0, 0, 0])
    >>> np.set_printoptions(suppress=True)
    >>> print(stft(signal, size=4, shift=2, fading=True, window=np.ones))
    [[ 0.+0.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  0.+0.j]
     [ 1.+0.j  0.+1.j -1.+0.j]
     [-2.+0.j  3.-1.j -4.+0.j]
     [ 2.+0.j -8.+0.j  2.+0.j]
     [ 5.+0.j  5.+0.j  5.+0.j]
     [ 0.+0.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  0.+0.j]]
    >>> activity_time_to_frequency(vad, stft_window_length=4, stft_shift=2, stft_fading=True)
    array([False, False,  True,  True,  True,  True, False, False])
    >>> activity_time_to_frequency([vad, vad], stft_window_length=4, stft_shift=2, stft_fading=True)
    array([[False, False,  True,  True,  True,  True, False, False],
           [False, False,  True,  True,  True,  True, False, False]])
    >>> print(stft(signal, size=4, shift=2, fading=False, window=np.ones))
    [[ 0.+0.j  0.+0.j  0.+0.j]
     [ 1.+0.j  0.+1.j -1.+0.j]
     [-2.+0.j  3.-1.j -4.+0.j]
     [ 2.+0.j -8.+0.j  2.+0.j]
     [ 5.+0.j  5.+0.j  5.+0.j]
     [ 0.+0.j  0.+0.j  0.+0.j]]
    >>> activity_time_to_frequency(vad, stft_window_length=4, stft_shift=2, stft_fading=False)
    array([False,  True,  True,  True,  True, False])
    >>> activity_time_to_frequency([vad, vad], stft_window_length=4, stft_shift=2, stft_fading=False)
    array([[False,  True,  True,  True,  True, False],
           [False,  True,  True,  True,  True, False]])


    >>> activity_time_to_frequency(np.zeros(200000), stft_window_length=1024, stft_shift=256, stft_fading=False, stft_pad=False).shape
    (778,)
    >>> from nara_wpe.utils import stft
    >>> stft(np.zeros(200000), size=1024, shift=256, fading=False, pad=False).shape
    (778, 513)
    """
    assert np.asarray(time_activity).dtype != np.object, (
        type(time_activity),
        np.asarray(time_activity).dtype,
    )
    time_activity = np.asarray(time_activity)

    if stft_fading:
        pad_width = np.array([(0, 0)] * time_activity.ndim)
        pad_width[-1, :] = stft_window_length - stft_shift  # Consider fading
        time_activity = np.pad(time_activity, pad_width, mode="constant")

    return segment_axis(
        time_activity,
        length=stft_window_length,
        shift=stft_shift,
        end="pad" if stft_pad else "cut",
    ).any(axis=-1)
