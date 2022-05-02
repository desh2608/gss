from dataclasses import dataclass

from gss.beamformer import beamform_mvdr


@dataclass
class Beamformer:
    postfilter: str

    def __call__(self, Obs, target_mask, distortion_mask):

        X_hat = beamform_mvdr(
            Y=Obs, X_mask=target_mask, N_mask=distortion_mask, ban=True
        )

        if self.postfilter is None:
            pass
        elif self.postfilter == "mask_mul":
            X_hat = X_hat * target_mask
        else:
            raise NotImplementedError(self.postfilter)

        return X_hat
