# The functions in this module are modified from:
# https://github.com/fgnt/nara_wpe/blob/master/nara_wpe/wpe.py

import functools
import operator

import cupy as cp
import numpy as np

from gss.utils.numpy_utils import segment_axis


def get_working_shape(shape):
    "Flattens all but the last two dimension."
    product = functools.reduce(operator.mul, [1] + list(shape[:-2]))
    return [product] + list(shape[-2:])


def _stable_solve(A, B):
    assert A.shape[:-2] == B.shape[:-2], (A.shape, B.shape)
    assert A.shape[-1] == B.shape[-2], (A.shape, B.shape)
    try:
        return cp.linalg.solve(A, B)
    except:
        shape_A, shape_B = A.shape, B.shape
        assert shape_A[:-2] == shape_A[:-2]
        working_shape_A = get_working_shape(shape_A)
        working_shape_B = get_working_shape(shape_B)
        A = A.reshape(working_shape_A)
        B = B.reshape(working_shape_B)

        C = cp.zeros_like(B)
        for i in range(working_shape_A[0]):
            # lstsq is much slower, use it only when necessary
            try:
                C[i] = cp.linalg.solve(A[i], B[i])
            except cp.linalg.linalg.LinAlgError:
                C[i] = cp.linalg.lstsq(A[i], B[i])[0]
        return C.reshape(*shape_B)


def build_y_tilde(Y, taps, delay):
    S = Y.shape[:-2]
    D = Y.shape[-2]
    T = Y.shape[-1]

    def pad(x, axis=-1, pad_width=taps + delay - 1):
        npad = np.zeros([x.ndim, 2], dtype=int)
        npad[axis, 0] = pad_width
        x = cp.pad(x, pad_width=npad, mode="constant", constant_values=0)
        return x

    Y_ = pad(Y)
    Y_ = cp.moveaxis(Y_, -1, -2)
    Y_ = cp.flip(Y_, axis=-1)
    Y_ = cp.ascontiguousarray(Y_)
    Y_ = cp.flip(Y_, axis=-1)
    Y_ = segment_axis(Y_, taps, 1, axis=-2)
    Y_ = cp.flip(Y_, axis=-2)
    if delay > 0:
        Y_ = Y_[..., :-delay, :, :]
    Y_ = cp.reshape(Y_, list(S) + [T, taps * D])
    Y_ = cp.moveaxis(Y_, -2, -1)

    return Y_


def hermite(x):
    return x.swapaxes(-2, -1).conj()


def get_power_inverse(signal, psd_context=0):
    power = cp.mean(abs_square(signal), axis=-2)

    if np.isposinf(psd_context):
        power = cp.broadcast_to(cp.mean(power, axis=-1, keepdims=True), power.shape)
    elif psd_context > 0:
        assert int(psd_context) == psd_context, psd_context
        psd_context = int(psd_context)
        power = window_mean(power, (psd_context, psd_context))
    elif psd_context == 0:
        pass
    else:
        raise ValueError(psd_context)
    return _stable_positive_inverse(power)


def abs_square(x):
    if cp.iscomplexobj(x):
        return x.real**2 + x.imag**2
    else:
        return x**2


def window_mean(x, lr_context, axis=-1):
    if isinstance(lr_context, int):
        lr_context = [lr_context + 1, lr_context]
    else:
        assert len(lr_context) == 2, lr_context
        tmp_l_context, tmp_r_context = lr_context
        lr_context = tmp_l_context + 1, tmp_r_context

    x = cp.asarray(x)

    window_length = sum(lr_context)
    if window_length == 0:
        return x

    pad_width = np.zeros((x.ndim, 2), dtype=int64)
    pad_width[axis] = lr_context

    first_slice = [slice(None)] * x.ndim
    first_slice[axis] = slice(sum(lr_context), None)
    second_slice = [slice(None)] * x.ndim
    second_slice[axis] = slice(None, -sum(lr_context))

    def foo(x):
        cumsum = cp.cumsum(cp.pad(x, pad_width, mode="constant"), axis=axis)
        return cumsum[first_slice] - cumsum[second_slice]

    ones_shape = [1] * x.ndim
    ones_shape[axis] = x.shape[axis]

    return foo(x) / foo(cp.ones(ones_shape, cp.int64))


def _stable_positive_inverse(power):
    eps = 1e-10 * cp.max(power)
    if eps == 0:
        # Special case when signal is zero.
        # Does not happen on real data.
        # This only happens in artificial cases, e.g. redacted signal parts,
        # where the signal is set to be zero from a human.
        #
        # The scale of the power does not matter, so take 1.
        inverse_power = cp.ones_like(power)
    else:
        cp.clip(power, a_min=eps, a_max=None, out=power)
        inverse_power = 1 / power
    return inverse_power


def wpe(Y, taps=10, delay=3, iterations=3, psd_context=0, statistics_mode="full"):
    """
    Batched WPE implementation (same as wpe_v6 in nara_wpe)

    Applicable in for-loops.

    Args:
        Y: Complex valued STFT signal with shape (F, D, T).
        taps: Filter order
        delay: Delay as a guard interval, such that X does not become zero.
        iterations:
        psd_context: Defines the number of elements in the time window
            to improve the power estimation. Total number of elements will
            be (psd_context + 1 + psd_context).
        statistics_mode: Either 'full' or 'valid'.
            'full': Pad the observation with zeros on the left for the
            estimation of the correlation matrix and vector.
            'valid': Only calculate correlation matrix and vector on valid
            slices of the observation.

    Returns:
        Estimated signal with the same shape as Y

    """

    if statistics_mode == "full":
        s = Ellipsis
    elif statistics_mode == "valid":
        s = (Ellipsis, slice(delay + taps - 1, None))
    else:
        raise ValueError(statistics_mode)

    X = cp.copy(Y)
    Y_tilde = build_y_tilde(Y, taps, delay)
    for iteration in range(iterations):
        inverse_power = get_power_inverse(X, psd_context=psd_context)
        Y_tilde_inverse_power = Y_tilde * inverse_power[..., None, :]
        R = cp.matmul(Y_tilde_inverse_power[s], hermite(Y_tilde[s]))
        P = cp.matmul(Y_tilde_inverse_power[s], hermite(Y[s]))
        G = _stable_solve(R, P)
        X = Y - cp.matmul(hermite(G), Y_tilde)

    return X
