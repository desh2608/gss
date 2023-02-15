# The functions here are modified from:
# https://github.com/fgnt/paderbox/blob/master/paderbox/transform/module_stft.py
import math
import string
import warnings
from typing import Optional, Union

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
            pad_width[axis, 1] = math.ceil((window_length - shift) / 2)
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
    fading: Optional[Union[bool, str]] = "full",
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
            ..., int(pad_width) : time_signal.shape[-1] - math.ceil(pad_width)
        ]

    return time_signal


# The following are modified from:
# https://github.com/pytorch/audio/blob/main/torchaudio/functional/functional.py


def _hz_to_mel(freq: float, mel_scale: str = "htk") -> float:
    r"""Convert Hz to Mels.
    Args:
        freqs (float): Frequencies in Hz
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)
    Returns:
        mels (float): Frequency in Mels
    """

    if mel_scale not in ["slaney", "htk"]:
        raise ValueError('mel_scale should be one of "htk" or "slaney".')

    if mel_scale == "htk":
        return 2595.0 * math.log10(1.0 + (freq / 700.0))

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (freq - f_min) / f_sp

    # Fill in the log-scale part
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    if freq >= min_log_hz:
        mels = min_log_mel + math.log(freq / min_log_hz) / logstep

    return mels


def _mel_to_hz(mels: cp.ndarray, mel_scale: str = "htk") -> cp.ndarray:
    """Convert mel bin numbers to frequencies.
    Args:
        mels (Tensor): Mel frequencies
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)
    Returns:
        freqs (Tensor): Mels converted in Hz
    """

    if mel_scale not in ["slaney", "htk"]:
        raise ValueError('mel_scale should be one of "htk" or "slaney".')

    if mel_scale == "htk":
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * cp.exp(logstep * (mels[log_t] - min_log_mel))

    return


def _create_triangular_filterbank(
    all_freqs: cp.ndarray,
    f_pts: cp.ndarray,
) -> cp.ndarray:
    """Create a triangular filter bank.
    Args:
        all_freqs (Array): STFT freq points of size (`n_freqs`).
        f_pts (Array): Filter mid points of size (`n_filter`).
    Returns:
        fb (Array): The filter bank of size (`n_freqs`, `n_filter`).
    """
    # Adopted from Librosa
    # calculate the difference between each filter mid point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_filter + 1)
    slopes = cp.expand_dims(f_pts, axis=0) - cp.expand_dims(
        all_freqs, axis=1
    )  # (n_freqs, n_filter + 2)
    # create overlapping triangles
    zero = cp.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_filter)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_filter)
    fb = cp.maximum(zero, cp.minimum(down_slopes, up_slopes))

    return fb


def mel_scale(
    n_freqs: int,
    n_mels: int,
    sample_rate: int,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
    norm: Optional[str] = None,
    mel_scale: str = "htk",
) -> cp.ndarray:
    r"""Create a frequency bin conversion matrix.
    Note:
        For the sake of the numerical compatibility with librosa, not all the coefficients
        in the resulting filter bank has magnitude of 1.
        .. image:: https://download.pytorch.org/torchaudio/doc-assets/mel_fbanks.png
           :alt: Visualization of generated filter bank
    Args:
        n_freqs (int): Number of frequencies to highlight/apply
        f_min (float): Minimum frequency (Hz)
        f_max (float): Maximum frequency (Hz)
        n_mels (int): Number of mel filterbanks
        sample_rate (int): Sample rate of the audio waveform
        norm (str or None, optional): If "slaney", divide the triangular mel weights by the width of the mel band
            (area normalization). (Default: ``None``)
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)
    Returns:
        Tensor: Triangular filter banks (fb matrix) of size (``n_freqs``, ``n_mels``)
        meaning number of frequencies to highlight/apply to x the number of filterbanks.
        Each column is a filterbank so that assuming there is a matrix A of
        size (..., ``n_freqs``), the applied result would be
        ``A * melscale_fbanks(A.size(-1), ...)``.
    """

    if norm is not None and norm != "slaney":
        raise ValueError('norm must be one of None or "slaney"')

    # freq bins
    all_freqs = cp.linspace(0, sample_rate // 2, n_freqs)

    f_max = f_max or float(sample_rate // 2)

    # calculate mel freq bins
    m_min = _hz_to_mel(f_min, mel_scale=mel_scale)
    m_max = _hz_to_mel(f_max, mel_scale=mel_scale)

    m_pts = cp.linspace(m_min, m_max, num=n_mels + 2)
    f_pts = _mel_to_hz(m_pts, mel_scale=mel_scale)

    # create filterbank
    fb = _create_triangular_filterbank(all_freqs, f_pts)

    if norm is not None and norm == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
        fb *= cp.expand_dims(enorm, axis=0)

    if (fb.max(axis=0) == 0.0).any():
        warnings.warn(
            "At least one mel filterbank has all zero values. "
            f"The value for `n_mels` ({n_mels}) may be set too high. "
            f"Or, the value for `n_freqs` ({n_freqs}) may be set too low."
        )

    return fb
