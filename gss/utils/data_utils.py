import numpy as np

from gss.utils.numpy_utils import segment_axis


def start_end_context_frames(start_context_samples, end_context_samples, stft_size, stft_shift, stft_fading):
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

                