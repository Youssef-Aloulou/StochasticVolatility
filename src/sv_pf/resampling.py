"""Resampling continu (Pitt 2002) : likelihood lisse en theta (tâche 2)."""
import numpy as np
import particles


def continuous_resample(v, w, u):
    """Resampling continu : inverse d'une CDF affine par morceaux.
    Partagé par le SV-GARCH maison ET ContinuousSMC."""
    idx = np.argsort(v); vs, ws = v[idx], w[idx]; M = len(v)
    gap = 0.5 * (ws[:-1] + ws[1:])
    C = np.empty(M); C[0] = 0.5 * ws[0]; C[1:] = C[0] + np.cumsum(gap)
    k = np.searchsorted(C, u, side="right") - 1
    kc = np.clip(k, 0, M - 2)
    frac = np.clip(np.divide(u - C[kc], gap[kc], out=np.zeros(M), where=gap[kc] > 0), 0, 1)
    out = vs[kc] + frac * (vs[kc + 1] - vs[kc])
    out[k < 0] = vs[0]; out[k >= M - 1] = vs[-1]
    return out

"""Resampling continu (Pitt 2002) : SV, SVL, SVLJ"""
class ContinuousSMC(particles.SMC):
    """SMC avec resampling continu (pour SV / SVL / SVLJ, état 1D).
    cont_u = uniformes triés fixés (continuité en theta)."""
    def __init__(self, *a, cont_u=None, **kw):
        super().__init__(*a, **kw)
        self.cont_u = cont_u

    def resample_move(self):
        self.rs_flag = True
        self.Xp = continuous_resample(self.X, self.aux.W, self.cont_u)
        self.A = np.arange(self.N)
        self.reset_weights()
        self.X = self.fk.M(self.t, self.Xp)