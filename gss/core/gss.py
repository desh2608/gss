from dataclasses import dataclass

import numpy as np

from pb_bss.distribution import CACGMMTrainer


@dataclass
class GSS:
    iterations: int
    iterations_post: int

    # use_pinv: bool = False
    # stable: bool = True

    def __call__(self, Obs, acitivity_freq):

        initialization = np.asarray(acitivity_freq, dtype=np.float64)
        initialization = np.where(initialization == 0, 1e-10, initialization)
        initialization = initialization / np.sum(initialization, keepdims=True, axis=0)
        initialization = np.repeat(initialization[None, ...], 513, axis=0)

        source_active_mask = np.asarray(acitivity_freq, dtype=np.bool)
        source_active_mask = np.repeat(source_active_mask[None, ...], 513, axis=0)


        cacGMM = CACGMMTrainer()

        all_affiliations = []
        F = Obs.shape[-1]
        T = Obs.T.shape[-2]
        for f in range(F):

            # T: Consider end of signal.
            # This should not be nessesary, but activity is for inear and not for
            # array.
            cur = cacGMM.fit(
                y=Obs.T[f, ...],
                initialization=initialization[f, ..., :T],
                iterations=self.iterations,
                source_activity_mask=source_active_mask[f, ..., :T],
                # return_affiliation=True,
            )

            if self.iterations_post != 0:
                if self.iterations_post != 1:
                    cur = cacGMM.fit(
                        y=Obs.T[f, ...],
                        initialization=cur,
                        iterations=self.iterations_post - 1,
                    )
                affiliation = cur.predict(
                    Obs.T[f, ...],
                )
            else:
                affiliation = cur.predict(
                    Obs.T[f, ...], source_activity_mask=source_active_mask[f, ..., :T]
                )

            all_affiliations.append(affiliation)

        posterior = np.array(all_affiliations).transpose(1, 2, 0)

        return posterior
