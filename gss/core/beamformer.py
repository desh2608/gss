from dataclasses import dataclass

import numpy as np


@dataclass
class Beamformer:
    type: str
    postfilter: str

    def __call__(self, Obs, target_mask, distortion_mask):
        bf = self.type

        if bf == "mvdrSouden_ban":
            from gss.beamform.beamforming_wrapper import (
                beamform_mvdr_souden_from_masks,
            )

            X_hat = beamform_mvdr_souden_from_masks(
                Y=Obs,
                X_mask=target_mask,
                N_mask=distortion_mask,
                ban=True,
            )
        elif bf == "ch2":
            X_hat = Obs[2]
        elif bf == "sum":
            X_hat = np.sum(Obs, axis=0)
        # elif bf is None:
        #     X_hat = Obs
        else:
            raise NotImplementedError(bf)

        if self.postfilter is None:
            pass
        elif self.postfilter == "mask_mul":
            X_hat = X_hat * target_mask
        else:
            raise NotImplementedError(self.postfilter)

        return X_hat
