from dataclasses import dataclass

import cupy as cp

from cacgmm.cacgmm_trainer import CACGMMTrainer


@dataclass
class GSS:
    iterations: int
    iterations_post: int

    def __call__(self, Obs, acitivity_freq):

        initialization = cp.asarray(acitivity_freq, dtype=cp.float64)
        initialization = cp.where(initialization == 0, 1e-10, initialization)
        initialization = initialization / cp.sum(initialization, keepdims=True, axis=0)
        initialization = cp.repeat(initialization[None, ...], 513, axis=0)

        source_active_mask = cp.asarray(acitivity_freq, dtype=cp.bool)
        source_active_mask = cp.repeat(source_active_mask[None, ...], 513, axis=0)

        cacGMM = CACGMMTrainer()

        D, T, F = Obs.shape

        cur = cacGMM.fit(
            y=Obs.T,
            initialization=initialization[..., :T],
            iterations=self.iterations,
            source_activity_mask=source_active_mask[..., :T],
        )

        if self.iterations_post != 0:
            if self.iterations_post != 1:
                cur = cacGMM.fit(
                    y=Obs.T,
                    initialization=cur,
                    iterations=self.iterations_post - 1,
                )
            affiliation = cur.predict(Obs.T)
        else:
            affiliation = cur.predict(
                Obs.T, source_activity_mask=source_active_mask[..., :T]
            )

        posterior = affiliation.transpose(1, 2, 0)

        return posterior
