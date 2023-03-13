# The functions here are modified from:
# https://github.com/fgnt/pb_bss/blob/master/pb_bss/distribution/cacgmm.py

from operator import xor

import cupy as cp

from gss.cacgmm.cacg_trainer import ComplexAngularCentralGaussianTrainer
from gss.cacgmm.cacgmm import CACGMM
from gss.cacgmm.utils import estimate_mixture_weight, normalize_observation


class CACGMMTrainer:
    def fit(
        self,
        y,
        initialization=None,
        num_classes=None,
        iterations=100,
        saliency=None,
        *,
        source_activity_mask=None,
        weight_constant_axis=(-1,),
        hermitize=True,
        covariance_norm="eigenvalue",
        affiliation_eps=1e-10,
        eigenvalue_floor=1e-10,
    ):
        """

        Args:
            y: Shape (frequency, time, channel) or (F, T, D)
            initialization:
                Affiliations between 0 and 1. Shape (F, K, T)
                or CACGMM instance
            num_classes: Scalar >0
            iterations: Scalar >0
            saliency:
                Importance weighting for each observation, shape (..., T)
                Should be pre-calculated externally, not just a string.
            source_activity_mask: Boolean mask that says for each time point
                for each source if it is active or not.
                Shape (F, K, T)
            weight_constant_axis: The axis that is used to calculate the mean
                over the affiliations. The affiliations have the
                shape (F, K, T), so the default value means averaging over
                the sample dimension. Note that averaging over an independent
                axis is supported.
            hermitize:
            covariance_norm: 'eigenvalue', 'trace' or False
            affiliation_eps:
            eigenvalue_floor: Relative flooring of the covariance eigenvalues

        Returns:

        """
        assert xor(initialization is None, num_classes is None), (
            "Incompatible input combination. "
            "Exactly one of the two inputs has to be None: "
            f"{initialization is None} xor {num_classes is None}"
        )

        assert cp.iscomplexobj(y), y.dtype
        assert y.shape[-1] > 1, y.shape
        y = normalize_observation(y)  # swap D and T dim, now y is F, D, T

        assert iterations > 0, iterations

        model = None

        *independent, D, num_observations = y.shape
        if initialization is None:
            assert num_classes is not None, num_classes
            affiliation_shape = (*independent, num_classes, num_observations)
            affiliation = cp.random.uniform(size=affiliation_shape)
            affiliation /= cp.einsum("fkn->fn", affiliation)[..., None, :]
            quadratic_form = cp.ones(affiliation_shape, dtype=y.real.dtype)
        elif isinstance(initialization, cp.ndarray):
            num_classes = initialization.shape[-2]
            assert num_classes > 1, num_classes
            affiliation_shape = (*independent, num_classes, num_observations)

            # Force same number of dims (Prevent wrong input)
            assert initialization.ndim == len(affiliation_shape), (
                initialization.shape,
                affiliation_shape,
            )

            # Allow singleton dimensions to be broadcasted
            assert initialization.shape[-2:] == affiliation_shape[-2:], (
                initialization.shape,
                affiliation_shape,
            )

            affiliation = cp.broadcast_to(initialization, affiliation_shape)
            quadratic_form = cp.ones(affiliation_shape, dtype=y.real.dtype)
        elif isinstance(initialization, CACGMM):
            # weight[-2] may be 1, when weight is fixed to 1/K
            # num_classes = initialization.weight.shape[-2]
            num_classes = initialization.cacg.covariance_eigenvectors.shape[-3]

            model = initialization
        else:
            raise TypeError("No sufficient initialization.")

        if source_activity_mask is not None:
            assert (
                source_activity_mask.dtype == bool
            ), source_activity_mask.dtype  # noqa
            assert source_activity_mask.shape[-2:] == (num_classes, num_observations), (
                source_activity_mask.shape,
                independent,
                num_classes,
                num_observations,
            )  # noqa

            if isinstance(initialization, cp.ndarray):
                assert source_activity_mask.shape == initialization.shape, (
                    source_activity_mask.shape,
                    initialization.shape,
                )  # noqa

        assert num_classes < 20, f"num_classes: {num_classes}, sure?"
        assert D < 35, f"Channels: {D}, sure?"

        for iteration in range(iterations):
            if model is not None:
                affiliation, quadratic_form, _ = model._predict(
                    y,
                    source_activity_mask=source_activity_mask,
                    affiliation_eps=affiliation_eps,
                )

            model = self._m_step(
                y,
                quadratic_form,
                affiliation=affiliation,
                saliency=saliency,
                hermitize=hermitize,
                covariance_norm=covariance_norm,
                eigenvalue_floor=eigenvalue_floor,
                weight_constant_axis=weight_constant_axis,
            )

        return model

    def fit_predict(
        self,
        y,
        initialization=None,
        num_classes=None,
        iterations=100,
        *,
        saliency=None,
        source_activity_mask=None,
        weight_constant_axis=(-1,),
        hermitize=True,
        covariance_norm="eigenvalue",
        affiliation_eps=1e-10,
        eigenvalue_floor=1e-10,
    ):
        """Fit a model. Then just return the posterior affiliations."""
        model = self.fit(
            y=y,
            initialization=initialization,
            num_classes=num_classes,
            iterations=iterations,
            saliency=saliency,
            source_activity_mask=source_activity_mask,
            weight_constant_axis=weight_constant_axis,
            hermitize=hermitize,
            covariance_norm=covariance_norm,
            affiliation_eps=affiliation_eps,
            eigenvalue_floor=eigenvalue_floor,
        )
        return model.predict(y)

    def _m_step(
        self,
        x,
        quadratic_form,
        affiliation,
        saliency,
        hermitize,
        covariance_norm,
        eigenvalue_floor,
        weight_constant_axis,
    ):
        weight = estimate_mixture_weight(
            affiliation=affiliation,
            saliency=saliency,
            weight_constant_axis=weight_constant_axis,
        )

        if saliency is None:
            masked_affiliation = affiliation
        else:
            masked_affiliation = affiliation * saliency[..., None, :]

        cacg = ComplexAngularCentralGaussianTrainer()._fit(
            y=x[..., None, :, :],
            saliency=masked_affiliation,
            quadratic_form=quadratic_form,
            hermitize=hermitize,
            covariance_norm=covariance_norm,
            eigenvalue_floor=eigenvalue_floor,
        )
        return CACGMM(weight=weight, cacg=cacg)
