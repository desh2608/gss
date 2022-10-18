# The functions here are modified from:
# https://github.com/fgnt/pb_bss/blob/master/pb_bss/distribution/cacgmm.py

from dataclasses import dataclass, field

import cupy as cp

from gss.cacgmm.cacg import ComplexAngularCentralGaussian
from gss.cacgmm.utils import log_pdf_to_affiliation, normalize_observation


def logsumexp(a, axis=None):
    a_max = cp.amax(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~cp.isfinite(a_max)] = 0
    elif not cp.isfinite(a_max):
        a_max = 0

    tmp = cp.exp(a - a_max)

    # suppress warnings about log of zero
    with cp.errstate(divide="ignore"):
        s = cp.sum(tmp, axis=axis, keepdims=False)
        out = cp.log(s)

    a_max = cp.squeeze(a_max, axis=axis)
    out += a_max

    return out


@dataclass
class CACGMM:
    weight: cp.array = None  # (..., K, 1) for weight_constant_axis==(-1,)  (..., 1, K, T) for weight_constant_axis==(-3,)
    cacg: ComplexAngularCentralGaussian = field(
        default_factory=ComplexAngularCentralGaussian
    )

    def predict(self, y, return_quadratic_form=False, source_activity_mask=None):
        assert cp.iscomplexobj(y), y.dtype
        y = normalize_observation(y)  # swap D and T dim
        affiliation, quadratic_form, _ = self._predict(
            y, source_activity_mask=source_activity_mask
        )
        if return_quadratic_form:
            return affiliation, quadratic_form
        else:
            return affiliation

    def _predict(self, y, source_activity_mask=None, affiliation_eps=0.0):
        """
        Note: y shape is (..., D, T) and not (..., T, D) like in predict
        Args:
            y: Normalized observations with shape (..., D, T).
        Returns: Affiliations with shape (..., K, T) and quadratic format
            with the same shape.
        """
        *independent, _, num_observations = y.shape

        log_pdf, quadratic_form = self.cacg._log_pdf(y[..., None, :, :])

        affiliation = log_pdf_to_affiliation(
            self.weight,
            log_pdf,
            source_activity_mask=source_activity_mask,
            affiliation_eps=affiliation_eps,
        )

        return affiliation, quadratic_form, log_pdf

    def log_likelihood(self, y):
        assert cp.iscomplexobj(y), y.dtype
        y = normalize_observation(y)  # swap D and T dim
        affiliation, quadratic_form, log_pdf = self._predict(y)
        return self._log_likelihood(y, log_pdf)

    def _log_likelihood(self, y, log_pdf):
        """
        Note: y shape is (..., D, T) and not (..., T, D) like in log_likelihood
        Args:
            y: Normalized observations with shape (..., D, T).
            log_pdf: shape (..., K, T)
        Returns:
            log_likelihood, scalar
        """
        *independent, channels, num_observations = y.shape

        # log_pdf.shape: *independent, speakers, num_observations

        # first: sum above the speakers
        # second: sum above time frequency in log domain
        log_likelihood = cp.sum(logsumexp(log_pdf, axis=-2))
        return log_likelihood
