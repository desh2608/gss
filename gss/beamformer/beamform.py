# The functions here are modified from the corresponding implementations in:
# https://github.com/fgnt/pb_bss/blob/master/pb_bss/extraction/beamformer.py

import functools
import operator
import cupy as cp
import numpy as np


__all__ = [
    "get_power_spectral_density_matrix",
    "get_mvdr_vector_souden",
    "blind_analytic_normalization",
    "apply_beamforming_vector",
]


def get_power_spectral_density_matrix(
    observation,
    mask=None,
    sensor_dim=-2,
    source_dim=-2,
    time_dim=-1,
    normalize=True,
):
    # ensure negative dim indexes
    sensor_dim, source_dim, time_dim = (
        d % observation.ndim - observation.ndim
        for d in (sensor_dim, source_dim, time_dim)
    )

    # ensure observation shape (..., sensors, frames)
    obs_transpose = [
        i for i in range(-observation.ndim, 0) if i not in [sensor_dim, time_dim]
    ] + [sensor_dim, time_dim]
    observation = observation.transpose(obs_transpose)

    if mask is None:
        psd = cp.einsum("...dt,...et->...de", observation, observation.conj())

        # normalize
        psd /= observation.shape[-1]

    else:
        # Unfortunately, this function changes `mask`.
        mask = cp.copy(mask)

        # normalize
        if mask.dtype == cp.bool:
            mask = cp.asfarray(mask)

        if normalize:
            mask /= cp.maximum(
                cp.sum(mask, axis=time_dim, keepdims=True),
                1e-10,
            )

        if mask.ndim + 1 == observation.ndim:
            mask = cp.expand_dims(mask, -2)
            psd = cp.einsum(
                "...dt,...et->...de",
                mask * observation,
                observation.conj(),
            )
        else:
            # ensure shape (..., sources, frames)
            mask_transpose = [
                i
                for i in range(-observation.ndim, 0)
                if i not in [source_dim, time_dim]
            ] + [source_dim, time_dim]
            mask = mask.transpose(mask_transpose)

            psd = cp.einsum(
                "...kt,...dt,...et->...kde", mask, observation, observation.conj()
            )

            if source_dim < -2:
                # Assume PSD shape (sources, ..., sensors, sensors) is desired
                psd = cp.rollaxis(psd, -3, source_dim % observation.ndim)

    return psd


def blind_analytic_normalization(vector, noise_psd_matrix):
    nominator = cp.einsum(
        "...a,...ab,...bc,...c->...",
        vector.conj(),
        noise_psd_matrix,
        noise_psd_matrix,
        vector,
    )
    nominator = cp.sqrt(nominator)

    denominator = cp.einsum(
        "...a,...ab,...b->...", vector.conj(), noise_psd_matrix, vector
    )
    denominator = cp.sqrt(denominator * denominator.conj())

    # We do the division in numpy since the `where` argument is not available in cupy
    nominator = cp.asnumpy(nominator)
    denominator = cp.asnumpy(denominator)
    normalization = np.divide(  # https://stackoverflow.com/a/37977222/5766934
        nominator, denominator, out=np.zeros_like(nominator), where=denominator != 0
    )
    normalization = cp.asarray(normalization)

    return vector * cp.abs(normalization[..., cp.newaxis])


def apply_beamforming_vector(vector, mix):
    assert vector.shape[-1] < 30, (vector.shape, mix.shape)
    return cp.einsum("...a,...at->...t", vector.conj(), mix)


def get_optimal_reference_channel(
    w_mat,
    target_psd_matrix,
    noise_psd_matrix,
    eps=None,
):
    if w_mat.ndim != 3:
        raise ValueError(
            "Estimating the ref_channel expects currently that the input "
            "has 3 ndims (frequency x sensors x sensors). "
            "Considering an independent dim in the SNR estimate is not "
            "unique."
        )
    if eps is None:
        eps = cp.finfo(w_mat.dtype).tiny
    SNR = cp.einsum(
        "...FdR,...FdD,...FDR->...R", w_mat.conj(), target_psd_matrix, w_mat
    ) / cp.maximum(
        cp.einsum("...FdR,...FdD,...FDR->...R", w_mat.conj(), noise_psd_matrix, w_mat),
        eps,
    )
    # Raises an exception when np.inf and/or np.NaN was in target_psd_matrix
    # or noise_psd_matrix
    assert cp.all(cp.isfinite(SNR)), SNR
    return cp.argmax(SNR.real)


def stable_solve(A, B):
    assert A.shape[:-2] == B.shape[:-2], (A.shape, B.shape)
    assert A.shape[-1] == B.shape[-2], (A.shape, B.shape)
    try:
        return cp.linalg.solve(A, B)
    except:  # noqa
        shape_A, shape_B = A.shape, B.shape
        assert shape_A[:-2] == shape_A[:-2]
        working_shape_A = [
            functools.reduce(operator.mul, [1, *shape_A[:-2]]),
            *shape_A[-2:],
        ]
        working_shape_B = [
            functools.reduce(operator.mul, [1, *shape_B[:-2]]),
            *shape_B[-2:],
        ]
        A = A.reshape(working_shape_A)
        B = B.reshape(working_shape_B)

        C = cp.zeros_like(B)
        for i in range(working_shape_A[0]):
            # lstsq is much slower, use it only when necessary
            try:
                C[i] = cp.linalg.solve(A[i], B[i])
            except cp.linalg.LinAlgError:
                C[i], *_ = cp.linalg.lstsq(A[i], B[i])
        return C.reshape(*shape_B)


def get_mvdr_vector_souden(
    target_psd_matrix,
    noise_psd_matrix,
    ref_channel=None,
    eps=None,
):
    assert noise_psd_matrix is not None

    phi = stable_solve(noise_psd_matrix, target_psd_matrix)
    lambda_ = cp.trace(phi, axis1=-1, axis2=-2)[..., None, None]
    if eps is None:
        eps = cp.finfo(lambda_.dtype).tiny
    mat = phi / cp.maximum(lambda_.real, eps)

    if ref_channel is None:
        ref_channel = get_optimal_reference_channel(
            mat, target_psd_matrix, noise_psd_matrix, eps=eps
        )

    beamformer = mat[..., ref_channel]
    return beamformer
