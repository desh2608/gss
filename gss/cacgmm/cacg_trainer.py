# The functions here are modified from:
# https://github.com/fgnt/pb_bss/blob/master/pb_bss/distribution/complex_angular_central_gaussian.py

import cupy as cp

from gss.cacgmm.cacg import ComplexAngularCentralGaussian
from gss.cacgmm.utils import (
    force_hermitian,
    is_broadcast_compatible,
    normalize_observation,
)


class ComplexAngularCentralGaussianTrainer:
    def fit(
        self,
        y,
        saliency=None,
        hermitize=True,
        covariance_norm="eigenvalue",
        eigenvalue_floor=1e-10,
        iterations=10,
    ):
        """
        Args:
            y: Should be normalized to unit norm. We normalize it anyway again.
               Shape (..., D, T), e.g. (1, D, T) for mixture models
            saliency: Shape (..., T), e.g. (K, T) for mixture models
            hermitize:
            eigenvalue_floor:
            iterations:
        Returns:
        """
        *independent, T, D = y.shape
        assert cp.iscomplexobj(y), y.dtype
        assert y.shape[-1] > 1
        y = normalize_observation(y)  # swap D and T dim

        if saliency is None:
            quadratic_form = cp.ones(*independent, T)
        else:
            raise NotImplementedError

        assert iterations > 0, iterations
        for _ in range(iterations):
            model = self._fit(
                y=y,
                saliency=saliency,
                quadratic_form=quadratic_form,
                hermitize=hermitize,
                covariance_norm=covariance_norm,
                eigenvalue_floor=eigenvalue_floor,
            )
            _, quadratic_form = model._log_pdf(y)

        return model

    def _fit(
        self,
        y,
        saliency,
        quadratic_form,
        hermitize=True,
        covariance_norm="eigenvalue",
        eigenvalue_floor=1e-10,
    ) -> ComplexAngularCentralGaussian:
        """Single step of the fit function. In general, needs iterations.
        Note: y shape is (..., D, T) and not (..., T, D) like in fit
        Args:
            y:  Assumed to have unit length.
                Shape (..., D, T), e.g. (1, D, T) for mixture models
            saliency: Shape (..., T), e.g. (K, T) for mixture models
            quadratic_form: (..., T), e.g. (K, T) for mixture models
            hermitize:
            eigenvalue_floor:
        """
        assert cp.iscomplexobj(y), y.dtype

        assert is_broadcast_compatible(y.shape[:-2], quadratic_form.shape[:-1]), (
            y.shape,
            quadratic_form.shape,
        )

        D = y.shape[-2]
        *independent, T = quadratic_form.shape

        if saliency is None:
            saliency = 1
            denominator = cp.array(T, dtype=cp.float64)
        else:
            assert y.ndim == saliency.ndim + 1, (y.shape, saliency.ndim)
            denominator = cp.einsum("...n->...", saliency)[..., None, None]

        # When the covariance matrix is zero, quadratic_form would also zero.
        # quadratic_form have to be positive
        quadratic_form = cp.maximum(
            quadratic_form,
            # Use 2 * tiny, because tiny is to small
            10 * cp.finfo(quadratic_form.dtype).tiny,
        )

        einsum_path = ["einsum_path", (0, 2), (0, 1)]
        covariance = D * cp.einsum(
            "...dn,...Dn,...n->...dD",
            y,
            y.conj(),
            (saliency / quadratic_form),
            optimize=einsum_path,
        )
        assert cp.isfinite(quadratic_form).all()
        covariance /= cp.maximum(
            denominator,
            cp.finfo(denominator.dtype).tiny,
        )
        assert covariance.shape == (*independent, D, D), (
            covariance.shape,
            (*independent, D, D),
        )

        assert cp.isfinite(covariance).all()

        if hermitize:
            covariance = force_hermitian(covariance)

        return ComplexAngularCentralGaussian.from_covariance(
            covariance,
            eigenvalue_floor=eigenvalue_floor,
            covariance_norm=covariance_norm,
        )
