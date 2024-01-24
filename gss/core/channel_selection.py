from dataclasses import dataclass

import cupy as cp
from lhotse.utils import EPSILON

from gss.core.stft_module import mel_scale


@dataclass
class EnvelopeVarianceChannelSelector:
    """
    Envelope Variance Channel Selection method.
    """

    n_mels: int = 40
    n_fft: int = 1024
    hop_length: int = 256
    sampling_rate: int = 16000
    chunk_size: float = 4
    chunk_stride: float = 2

    def __post_init__(self):
        self.subband_weights = cp.ones(self.n_mels)
        self.chunk_size = int(self.chunk_size * self.sampling_rate / self.hop_length)
        self.chunk_stride = int(
            self.chunk_stride * self.sampling_rate / self.hop_length
        )
        self.fb = mel_scale(
            n_freqs=self.n_fft, n_mels=self.n_mels, sample_rate=self.sampling_rate
        )

    def _single_window(self, mels):
        logmels = cp.log(mels + EPSILON)
        mels = cp.exp(logmels - cp.mean(logmels, axis=-1, keepdims=True))
        var = cp.var(mels ** (1 / 3), axis=-1)  # channels, subbands
        var = var / cp.amax(var, axis=1, keepdims=True)
        subband_weights = cp.abs(self.subband_weights)
        ranking = cp.sum(var * subband_weights, axis=-1)
        return ranking

    def _count_chunks(self, inlen, chunk_size, chunk_stride):
        return int((inlen - chunk_size + chunk_stride) / chunk_stride)

    def _get_chunks_indx(self, in_len, chunk_size, chunk_stride, discard_last=False):
        i = -1
        for i in range(self._count_chunks(in_len, chunk_size, chunk_stride)):
            yield i * chunk_stride, i * chunk_stride + chunk_size
        if not discard_last and i * chunk_stride + chunk_size < in_len:
            if in_len - (i + 1) * chunk_stride > 0:
                yield (i + 1) * chunk_stride, in_len

    def __call__(self, obs, num_channels):
        """
        Args:
            obs: (channels, time, freq)
        """
        assert obs.ndim == 3

        # (..., time, freq) dot (freq, n_mels) -> (..., n_mels, time)
        mels = cp.matmul(obs, self.fb).transpose(0, 2, 1)

        if mels.shape[-1] > (self.chunk_size + self.chunk_stride):
            # using for because i am too lazy of taking care of padded
            # values in stats computation, but this is fast

            indxs = self._get_chunks_indx(
                mels.shape[-1], self.chunk_size, self.chunk_stride
            )
            all_win_ranks = [self._single_window(mels[..., s:t]) for s, t in indxs]

            scores = cp.stack(all_win_ranks, axis=0).mean(axis=0)
        else:
            scores = self._single_window(mels)

        channel_ranks = cp.argsort(scores)[::-1]
        selected_channels = channel_ranks[:num_channels]
        return obs[selected_channels], selected_channels
