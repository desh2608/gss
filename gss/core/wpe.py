from dataclasses import dataclass
from nara_wpe.wpe import wpe_v8

from gss.utils.numpy_utils import morph


@dataclass
class WPE:
    taps: int
    delay: int
    iterations: int
    psd_context: int

    def __call__(self, Obs, stack=None):

        if Obs.ndim == 3:
            assert stack is None, stack
            Obs = wpe_v8(
                Obs.transpose(2, 0, 1),
                taps=self.taps,
                delay=self.delay,
                iterations=self.iterations,
                psd_context=self.psd_context,
            ).transpose(1, 2, 0)
        elif Obs.ndim == 4:
            if stack is True:
                _A = Obs.shape[0]
                Obs = morph("ACTF->A*CTF", Obs)
                Obs = wpe_v8(
                    Obs.transpose(2, 0, 1),
                    taps=self.taps,
                    delay=self.delay,
                    iterations=self.iterations,
                    psd_context=self.psd_context,
                ).transpose(1, 2, 0)
                Obs = morph("A*CTF->ACTF", Obs, A=_A)
            elif stack is False:
                Obs = wpe_v8(
                    Obs.transpose(0, 3, 1, 2),
                    taps=self.taps,
                    delay=self.delay,
                    iterations=self.iterations,
                    psd_context=self.psd_context,
                ).transpose(0, 2, 3, 1)

            else:
                raise NotImplementedError(stack)
        else:
            raise NotImplementedError(Obs.shape)

        return Obs
