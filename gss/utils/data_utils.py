import numpy as np

from gss.utils.numpy_utils import segment_axis
from gss.utils import keys as K


def start_end_context_frames(ex, stft_size, stft_shift, stft_fading):
    start_context_samples = ex["start_orig"] - ex["start"]
    end_context_samples = ex["end"] - ex["end_orig"]

    assert start_context_samples >= 0, (start_context_samples, ex)
    assert end_context_samples >= 0, (end_context_samples, ex)

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


def nest_map_structure(
    func,
    *structures,
    mapping_type=dict,
    sequence_type=(tuple, list),
):
    """

    Calls func(element) on each element of structure.
    See tensorflow.nest.map_structure.

    Args:
        func:
        structure: nested structure

    Returns:


    >>> structure = {'a': [1, 2, (3, 4)], 'b': [5, (6,)]}
    >>> nest_map_structure(lambda e: e + 10, structure)
    {'a': [11, 12, (13, 14)], 'b': [15, (16,)]}
    >>> nest_map_structure(lambda e: e + 10, {'a': 11, 'b': 12})
    {'a': 21, 'b': 22}
    >>> nest_map_structure(lambda e: e + 10, {'a': 11, 'b': [13, 14]})
    {'a': 21, 'b': [23, 24]}
    >>> nest_map_structure(lambda e: e * 2, structure, sequence_type=None)
    {'a': [1, 2, (3, 4), 1, 2, (3, 4)], 'b': [5, (6,), 5, (6,)]}

    >>> nest_map_structure(lambda a, b: a + b, structure, structure)
    {'a': [2, 4, (6, 8)], 'b': [10, (12,)]}


    >>> nest_map_structure(lambda a, b: a + b, structure, {'a': 2, 'b': 4})
    Traceback (most recent call last):
    ...
    AssertionError: ([<class 'list'>, <class 'int'>], ([1, 2, (3, 4)], 2))
    """
    types = {type(s) for s in structures}

    if mapping_type and isinstance(structures[0], mapping_type):
        assert len(types) == 1, ([type(s) for s in structures], structures)
        return structures[0].__class__(
            {
                k: nest_map_structure(
                    func,
                    *[s[k] for s in structures],
                    mapping_type=mapping_type,
                    sequence_type=sequence_type,
                )
                for k in structures[0].keys()
            }
        )
    elif sequence_type and isinstance(structures[0], sequence_type):
        assert len(types) == 1, ([type(s) for s in structures], structures)
        return structures[0].__class__(
            [
                nest_map_structure(
                    func, *args, mapping_type=mapping_type, sequence_type=sequence_type
                )
                for args in zip(*structures)
            ]
        )
    else:
        return func(*structures)


def backup_orig_start_end(ex):
    ex["start_orig"] = ex[K.START]
    ex["end_orig"] = ex[K.END]
    ex["num_samples_orig"] = ex[K.NUM_SAMPLES]
    return ex


def add_context(ex, samples, equal_start_context=False):

    start_context = end_context = samples

    assert "start_orig" in ex, ex
    assert "end_orig" in ex, ex
    assert "num_samples_orig" in ex, ex

    ex[K.START] = nest_map_structure(
        lambda time: max(time - start_context, 0),
        ex[K.START],
    )
    if equal_start_context:
        start_context_delta = nest_map_structure(
            lambda start, start_orig: start_orig - start,
            ex[K.START],
            ex["start_orig"],
        )

        start_context_delta_flat = []
        _ = nest_map_structure(
            lambda val: start_context_delta_flat.append(val),
            start_context_delta,
        )
        smallest_start_context = np.min(start_context_delta_flat)

        ex[K.START] = nest_map_structure(
            lambda time: max(time - smallest_start_context, 0),
            ex["start_orig"],
        )

    ex[K.END] = nest_map_structure(
        lambda time: time + end_context,
        ex[K.END],
    )

    ex[K.NUM_SAMPLES] = nest_map_structure(
        lambda start, end: end - start,
        ex[K.START],
        ex[K.END],
    )
    return ex
