# The functions here are modified from:
# https://github.com/fgnt/paderbox/blob/master/paderbox/transform/module_stft.py
import string
import typing
from math import ceil

import cupy as cp
import cupyx as cpx
import numpy as np
from cupy.fft import irfft, rfft

from gss.utils.numpy_utils import roll_zeropad, segment_axis


def stft(
    time_signal,
    size: int = 1024,
    shift: int = 256,
    *,
    axis=-1,
    fading=True,
) -> cp.array:
    """
    Calculates the short time Fourier transformation of a multi channel multi
    speaker time signal. It is able to add additional zeros for fade-in and
    fade out and should yield an STFT signal which allows perfect
    reconstruction.

    :param time_signal: Multi channel time signal with dimensions
        AA x ... x AZ x T x BA x ... x BZ.
    :param size: Scalar FFT-size.
    :param shift: Scalar FFT-shift, the step between successive frames in
        samples. Typically shift is a fraction of size.
    :param axis: Scalar axis of time.
        Default: None means the biggest dimension.
    :param fading: Pads the signal with zeros for better reconstruction.
    :return: Single channel complex STFT signal with dimensions
        AA x ... x AZ x T' times size/2+1 times BA x ... x BZ.
    """
    ndim = time_signal.ndim
    axis = axis % ndim

    window_length = size

    # Pad with zeros to have enough samples for the window function to fade.
    assert fading in [None, True, False, "full", "half"], fading
    if fading not in [False, None]:
        pad_width = np.zeros([ndim, 2], dtype=np.int)
        if fading == "half":
            pad_width[axis, 0] = (window_length - shift) // 2
            pad_width[axis, 1] = ceil((window_length - shift) / 2)
        else:
            pad_width[axis, :] = window_length - shift
        time_signal = cp.pad(time_signal, pad_width, mode="constant")

    window = cp.blackman(window_length + 1)[:-1]

    time_signal_seg = segment_axis(
        time_signal, window_length, shift=shift, axis=axis, end="pad"
    )

    letters = string.ascii_lowercase[: time_signal_seg.ndim]
    mapping = letters + "," + letters[axis + 1] + "->" + letters

    try:
        return rfft(
            cp.einsum(mapping, time_signal_seg, window),
            n=size,
            axis=axis + 1,
        )
    except ValueError as e:
        raise ValueError(
            f"Could not calculate the stft, something does not match.\n"
            f"mapping: {mapping}, "
            f"time_signal_seg.shape: {time_signal_seg.shape}, "
            f"window.shape: {window.shape}, "
            f"size: {size}"
            f"axis+1: {axis+1}"
        ) from e


def _biorthogonal_window_brute_force(analysis_window, shift, use_amplitude=False):
    """
    The biorthogonal window (synthesis_window) must verify the criterion:
        synthesis_window * analysis_window plus it's shifts must be one.
        1 == sum m from -inf to inf over (synthesis_window(n - mB) * analysis_window(n - mB))
        B ... shift
        n ... time index
        m ... shift index

    :param analysis_window:
    :param shift:
    :return:

    >>> analysis_window = signal.windows.blackman(4+1)[:-1]
    >>> print(analysis_window)
    [-1.38777878e-17  3.40000000e-01  1.00000000e+00  3.40000000e-01]
    >>> synthesis_window = _biorthogonal_window_brute_force(analysis_window, 1)
    >>> print(synthesis_window)
    [-1.12717575e-17  2.76153346e-01  8.12215724e-01  2.76153346e-01]
    >>> mult = analysis_window * synthesis_window
    >>> sum(mult)
    1.0000000000000002
    """
    size = len(analysis_window)

    influence_width = (size - 1) // shift

    denominator = cp.zeros_like(analysis_window)

    if use_amplitude:
        analysis_window_square = analysis_window
    else:
        analysis_window_square = analysis_window**2
    for i in range(-influence_width, influence_width + 1):
        denominator += roll_zeropad(analysis_window_square, shift * i)

    if use_amplitude:
        synthesis_window = 1 / denominator
    else:
        synthesis_window = analysis_window / denominator
    return synthesis_window


def istft(
    stft_signal,
    size: int = 1024,
    shift: int = 256,
    *,
    fading: typing.Optional[typing.Union[bool, str]] = "full",
):
    """
    Calculated the inverse short time Fourier transform to exactly reconstruct
    the time signal.

    ..note::
        Be careful if you make modifications in the frequency domain (e.g.
        beamforming) because the synthesis window is calculated according to
        the unmodified! analysis window.

    :param stft_signal: Single channel complex STFT signal
        with dimensions (..., frames, size/2+1).
    :param size: Scalar FFT-size.
    :param shift: Scalar FFT-shift. Typically shift is a fraction of size.
    :param fading: Removes the additional padding, if done during STFT.

    :return: Single channel complex STFT signal
    :return: Single channel time signal.
    """
    assert stft_signal.shape[-1] == size // 2 + 1, str(stft_signal.shape)

    window_length = size

    window = cp.blackman(window_length + 1)[:-1]
    window = _biorthogonal_window_brute_force(window, shift)

    # In the following, we use numpy.add.at since cupyx.scatter_add does not seem to be
    # giving the same results. We should replace this with cupy.add.at once it is
    # available in the stable release (see: https://github.com/cupy/cupy/pull/7077).

    time_signal = np.zeros(
        (*stft_signal.shape[:-2], stft_signal.shape[-2] * shift + window_length - shift)
    )

    # Get the correct view to time_signal
    time_signal_seg = segment_axis(time_signal, window_length, shift, end=None)

    np.add.at(
        time_signal_seg,
        ...,
        (window * cp.real(irfft(stft_signal, n=size))[..., :window_length]).get(),
    )
    # The [..., :window_length] is the inverse of the window padding in rfft.

    # Compensate fade-in and fade-out

    assert fading in [None, True, False, "full", "half"], fading
    if fading not in [None, False]:
        pad_width = window_length - shift
        if fading == "half":
            pad_width /= 2
        time_signal = time_signal[
            ..., int(pad_width) : time_signal.shape[-1] - ceil(pad_width)
        ]

    return time_signal
