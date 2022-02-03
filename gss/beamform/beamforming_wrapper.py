import numpy as np

from cached_property import cached_property

from pb_bss.extraction import beamformer
from pb_bss.extraction.mask_module import lorenz_mask, quantile_mask

from gss.utils.numpy_utils import morph


class _Beamformer:
    def __init__(
        self,
        Y,
        X_mask,
        N_mask,
        debug=False,
    ):
        self.debug = debug

        if np.ndim(Y) == 4:
            self.Y = morph("1DTF->FDT", Y)
        else:
            self.Y = morph("DTF->FDT", Y)

        if np.ndim(X_mask) == 4:
            self.X_mask = morph("1DTF->FT", X_mask, reduce=np.median)
            self.N_mask = morph("1DTF->FT", N_mask, reduce=np.median)
        elif np.ndim(X_mask) == 3:
            self.X_mask = morph("DTF->FT", X_mask, reduce=np.median)
            self.N_mask = morph("DTF->FT", N_mask, reduce=np.median)
        elif np.ndim(X_mask) == 2:
            self.X_mask = morph("TF->FT", X_mask, reduce=np.median)
            self.N_mask = morph("TF->FT", N_mask, reduce=np.median)
        else:
            raise NotImplementedError(X_mask.shape)

        if self.debug:
            print("Y", repr(self.Y))
            print("X_mask", repr(self.X_mask), "N_mask", repr(self.N_mask))

        assert self.Y.ndim == 3, self.Y.shape
        F, D, T = self.Y.shape
        assert D < 30, (D, self.Y.shape)
        assert self.X_mask.shape == (F, T), (self.X_mask.shape, F, T)
        assert self.N_mask.shape == (F, T), (self.N_mask.shape, F, T)

    @cached_property
    def _Cov_X(self):
        Cov_X = beamformer.get_power_spectral_density_matrix(self.Y, self.X_mask)
        if self.debug:
            print("Cov_X", repr(Cov_X))
        return Cov_X

    @cached_property
    def _Cov_N(self):
        Cov_N = beamformer.get_power_spectral_density_matrix(self.Y, self.N_mask)
        if self.debug:
            print("Cov_N", repr(Cov_N))
        return Cov_N

    @cached_property
    def _w_mvdr_souden(self):
        w_mvdr_souden = beamformer.get_mvdr_vector_souden(
            self._Cov_X, self._Cov_N, eps=1e-10
        )
        if self.debug:
            print("w_mvdr_souden", repr(w_mvdr_souden))
        return w_mvdr_souden

    @cached_property
    def _w_mvdr_souden_ban(self):
        w_mvdr_souden_ban = beamformer.blind_analytic_normalization(
            self._w_mvdr_souden, self._Cov_N
        )
        if self.debug:
            print("w_mvdr_souden_ban", repr(w_mvdr_souden_ban))
        return w_mvdr_souden_ban

    @cached_property
    def _w_gev(self):
        w_gev = beamformer.get_gev_vector(self._Cov_X, self._Cov_N, force_cython=True)
        if self.debug:
            print("w_gev", repr(w_gev))
        return w_gev

    @cached_property
    def _w_gev_ban(self):
        w_gev_ban = beamformer.blind_analytic_normalization(self._w_gev, self._Cov_N)
        if self.debug:
            print("w_gev_ban", repr(w_gev_ban))
        return w_gev_ban

    @cached_property
    def X_hat_mvdr_souden(self):
        return beamformer.apply_beamforming_vector(self._w_mvdr_souden, self.Y).T

    @cached_property
    def X_hat_mvdr_souden_ban(self):
        return beamformer.apply_beamforming_vector(self._w_mvdr_souden_ban, self.Y).T

    @cached_property
    def X_hat_gev(self):
        return beamformer.apply_beamforming_vector(self._w_gev, self.Y).T

    @cached_property
    def X_hat_gev_ban(self):
        return beamformer.apply_beamforming_vector(self._w_gev_ban, self.Y).T


def beamform_mvdr_souden_from_masks(
    Y,
    X_mask,
    N_mask,
    ban=False,
    debug=False,
):
    bf = _Beamformer(
        Y=Y,
        X_mask=X_mask,
        N_mask=N_mask,
        debug=debug,
    )
    if ban:
        return bf.X_hat_mvdr_souden_ban
    else:
        return bf.X_hat_mvdr_souden


def beamform_lcmv_souden_from_masks(
    Y,
    X_mask,
    I_mask,
    N_mask,
    ban=False,
    debug=False,
):
    if np.ndim(Y) == 4:
        Y = morph("1DTF->FDT", Y)
    else:
        Y = morph("DTF->FDT", Y)

    if np.ndim(X_mask) == 4:
        X_mask = morph("1DTF->FT", X_mask, reduce=np.median)
        I_mask = morph("1DTF->FT", I_mask, reduce=np.median)
        N_mask = morph("1DTF->FT", N_mask, reduce=np.median)
    elif np.ndim(X_mask) == 3:
        X_mask = morph("DTF->FT", X_mask, reduce=np.median)
        I_mask = morph("DTF->FT", I_mask, reduce=np.median)
        N_mask = morph("DTF->FT", N_mask, reduce=np.median)
    elif np.ndim(X_mask) == 2:
        X_mask = morph("TF->FT", X_mask, reduce=np.median)
        I_mask = morph("TF->FT", I_mask, reduce=np.median)
        N_mask = morph("TF->FT", N_mask, reduce=np.median)
    else:
        raise NotImplementedError(X_mask.shape)

    Cov_X = beamformer.get_power_spectral_density_matrix(Y, X_mask)
    Cov_I = beamformer.get_power_spectral_density_matrix(Y, I_mask)
    Cov_N = beamformer.get_power_spectral_density_matrix(Y, N_mask)

    w_lcmv_souden = beamformer.get_lcmv_vector_souden(Cov_X, Cov_I, Cov_N)
    if ban:
        w_lcmv_souden_ban = beamformer.blind_analytic_normalization(
            w_lcmv_souden, Cov_I + Cov_N
        )
        w = w_lcmv_souden_ban
    else:
        w = w_lcmv_souden

    return beamformer.apply_beamforming_vector(w, Y).T


def beamform_gev_from_masks(
    Y,
    X_mask,
    N_mask,
    ban=True,
    debug=False,
):
    bf = _Beamformer(
        Y=Y,
        X_mask=X_mask,
        N_mask=N_mask,
        debug=debug,
    )
    if ban:
        return bf.X_hat_gev_ban
    else:
        return bf.X_hat_gev


def beamform_mvdr_souden_with_lorenz_mask(
    Y,
    X_hat=None,
    debug=False,
):
    if X_hat is None:
        X_hat = Y

    X_mask = np.swapaxes(lorenz_mask(np.swapaxes(X_hat, -2, -1)), -2, -1)
    N_mask = 1 - X_mask

    return beamform_mvdr_souden_from_masks(
        Y=Y,
        X_mask=X_mask,
        N_mask=N_mask,
        debug=debug,
    )


def beamform_mvdr_souden_with_quantil_mask(
    Y,
    X_hat=None,
    debug=False,
    quantil=[0.1, -0.8],
):
    if X_hat is None:
        X_hat = Y

    X_mask, N_mask = quantile_mask(
        X_hat,
        quantile=quantil,
        sensor_axis=None,
        axis=-2,
    )

    return beamform_mvdr_souden_from_masks(
        Y=Y,
        X_mask=X_mask,
        N_mask=N_mask,
        debug=debug,
    )
