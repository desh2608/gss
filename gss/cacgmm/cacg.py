# The functions here are modified from:
# https://github.com/fgnt/pb_bss/blob/master/pb_bss/distribution/complex_angular_central_gaussian.py

from dataclasses import dataclass

import cupy as cp

from gss.cacgmm.utils import normalize_observation, is_broadcast_compatible


@dataclass
class ComplexAngularCentralGaussian:
    """
    Note:
        Instead of the covariance the eigenvectors and eigenvalues are saved.
        These saves some computations, because to have a more stable covariance,
        the eigenvalues are floored.
    """

    covariance_eigenvectors: cp.array = None  # (..., D, D)
    covariance_eigenvalues: cp.array = None  # (..., D)

    @classmethod
    def from_covariance(
        cls,
        covariance,
        eigenvalue_floor=0.0,
        covariance_norm="eigenvalue",
    ):
        if covariance_norm == "trace":
            cov_trace = cp.einsum("...dd", covariance)[..., None, None]
            covariance /= cp.maximum(cov_trace, cp.finfo(cov_trace.dtype).tiny)
        else:
            assert covariance_norm in ["eigenvalue", False]

        try:
            eigenvals, eigenvecs = cp.linalg.eigh(covariance)
        except cp.linalg.LinAlgError:
            # ToDo: figure out when this happen and why eig may work.
            # It is likely that eig is more stable than eigh.
            try:
                eigenvals, eigenvecs = cp.linalg.eig(covariance)
            except cp.linalg.LinAlgError:
                if eigenvalue_floor == 0:
                    raise RuntimeError(
                        "When you set the eigenvalue_floor to zero it can "
                        "happen that the eigenvalues get zero and the "
                        "reciprocal eigenvalue that is used in "
                        f"{cls.__name__}._log_pdf gets infinity."
                    )
                else:
                    raise
        eigenvals = eigenvals.real
        if covariance_norm == "eigenvalue":
            # The scale of the eigenvals does not matter.
            eigenvals = eigenvals / cp.maximum(
                cp.amax(eigenvals, axis=-1, keepdims=True),
                cp.finfo(eigenvals.dtype).tiny,
            )
            eigenvals = cp.maximum(
                eigenvals,
                eigenvalue_floor,
            )
        else:
            eigenvals = cp.maximum(
                eigenvals,
                cp.amax(eigenvals, axis=-1, keepdims=True) * eigenvalue_floor,
            )
        assert cp.isfinite(eigenvals).all(), eigenvals

        return cls(
            covariance_eigenvalues=eigenvals,
            covariance_eigenvectors=eigenvecs,
        )

    @property
    def covariance(self):
        return cp.einsum(
            "...wx,...x,...zx->...wz",
            self.covariance_eigenvectors,
            self.covariance_eigenvalues,
            self.covariance_eigenvectors.conj(),
            optimize="greedy",
        )

    @property
    def log_determinant(self):
        return cp.sum(cp.log(self.covariance_eigenvalues), axis=-1)

    def log_pdf(self, y):
        """
        Args:
            y: Shape (..., T, D)
        Returns:
        """
        y = normalize_observation(y)  # swap D and T dim
        log_pdf, _ = self._log_pdf(y)
        return log_pdf

    def _log_pdf(self, y):
        """Gets used by. e.g. the cACGMM.
        TODO: quadratic_form might be useful by itself
        Note: y shape is (..., D, T) and not (..., T, D) like in log_pdf
        Args:
            y: Normalized observations with shape (..., D, T).
        Returns: Affiliations with shape (..., K, T) and quadratic format
            with the same shape.
        """
        *independent, D, T = y.shape

        assert is_broadcast_compatible(
            [*independent, D, D], self.covariance_eigenvectors.shape
        ), (y.shape, self.covariance_eigenvectors.shape)

        quadratic_form = cp.maximum(
            cp.abs(
                cp.einsum(
                    "...dt,...de,...e,...ge,...gt->...t",
                    y.conj(),
                    self.covariance_eigenvectors,
                    1 / self.covariance_eigenvalues,
                    self.covariance_eigenvectors.conj(),
                    y,
                    optimize="optimal",
                )
            ),
            cp.finfo(y.dtype).tiny,
        )
        log_pdf = -D * cp.log(quadratic_form)
        log_pdf -= self.log_determinant[..., None]

        return log_pdf, quadratic_form
