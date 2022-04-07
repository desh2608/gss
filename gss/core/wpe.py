from dataclasses import dataclass
from wpe import wpe


@dataclass
class WPE:
    taps: int
    delay: int
    iterations: int
    psd_context: int

    def __call__(self, Obs):

        Obs = wpe(
            Obs.transpose(2, 0, 1),
            taps=self.taps,
            delay=self.delay,
            iterations=self.iterations,
            psd_context=self.psd_context,
        ).transpose(1, 2, 0)

        return Obs
