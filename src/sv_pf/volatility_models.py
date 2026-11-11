import warnings; warnings.simplefilter('ignore')  # hide warnings

# standard libraries
import numpy as np
import seaborn as sb

# modules from particles
import particles  # core module
from particles import distributions as dists  # where probability distributions are defined
from particles import state_space_models as ssm  # where state-space models are defined
from particles.collectors import Moments


class StochVolLeverageShifted(ssm.StochVol):
    r"""Univariate stochastic volatility model with leverage effect (paper timing).
 
    .. math::
 
        X_0                         & \sim N(\mu, \sigma^2/(1-\rho^2))     \\
        X_t|X_{t-1}=x_{t-1}         & \sim N(\mu + \rho (x_{t-1}-\mu), \sigma^2) \\
        Y_t|X_{t-1:t} =x_{t-1:t}    & \sim N( s \phi u, s^2 (1-\phi^2) )
 
    with :math:`s=\exp(x_{t-1}/2), u = [x_t-\mu-\rho*(x_{t-1}-\mu)]/\sigma`
 
    Note
    ----
 
    This is equivalent to assuming that the noise of Y_t is correlated, with
    correlation :math:`\phi`, with the innovation of the *next* state (shifted
    timing of Pitt, Malik & Doucet, 2014): the filter state is x_t = h_{t+1},
 
    .. math::
 
        X_t & = \mu + \rho(X_{t-1}-\mu) + \sigma U_t \\
        Y_t & = \exp(X_{t-1}/2) * V_t
 
    and :math:`Cor(U_t, V_t) = \phi`. Unlike StochVolLeverage, Y_t is scaled by
    the *previous* volatility and is centred conditionally (martingale
    difference). rho = persistence (phi in the paper), phi = leverage (rho in
    the paper), sigma = sigma_eta.
 
    Note
    ----
    At t=0 there is no previous state, so Y_0 falls back to the plain SV law
    N(0, e^{x_0}): one observation out of T, O(1/T) edge effect. Data is passed
    as is (y_0, ..., y_{T-1}); methods proposal and logeta are redefined
    (exact) and proposal0 is inherited from StochVol (valid, t=0 is plain SV).
    """
    default_params = {"mu": -1.02, "rho": 0.9702, "sigma": 0.178, "phi": 0.0}
 
    def PY(self, t, xp, x):
        # densité d'observation p(y_t | x_{t-1}, x_t)
        if t == 0:
            return super().PY(t, xp, x)
        u = (x - self.EXt(xp)) / self.sigma
        s = np.exp(0.5 * xp)
        return dists.Normal(loc=s * self.phi * u, scale=s * np.sqrt(1.0 - self.phi ** 2))

 
    def proposal(self, t, xp, data):
        eps = data[t] * np.exp(-0.5 * xp)          # = y_t e^{-h_t/2}  (etape 1a, sans saut)
        return dists.Normal(loc=self.EXt(xp) + self.sigma * self.phi * eps,   # etape 1b
                            scale=self.sigma * np.sqrt(1.0 - self.phi ** 2))
 
    def logeta(self, t, x, data):
        # fonction auxiliaire APF : log p(y_{t+1} | x_t)
        return dists.Normal(loc=0.0, scale=np.exp(0.5 * x)).logpdf(data[t + 1])

class StochVolLeverageJump(StochVolLeverageShifted):
    # kappa = P(J_t = 1), sigma_J = écart-type du saut
    default_params = {"mu": -1.02, "rho": 0.9702, "sigma": 0.178, "phi": 0.0,
                      "kappa": 0.02, "sigma_J": 3.0}

    def PY(self, t, xp, x):
        # mélange : (1-kappa) sans saut + kappa avec saut (variance + sigma_J^2)
        base = super().PY(t, xp, x)
        jump = dists.Normal(loc=base.loc,
                            scale=np.sqrt(base.scale ** 2 + self.sigma_J ** 2))
        return dists.Mixture([1.0 - self.kappa, self.kappa], base, jump)
    
    def logeta(self, t, x, data):
        s2 = np.exp(x)
        mix = dists.Mixture([1 - self.kappa, self.kappa],
                            dists.Normal(scale=np.sqrt(s2)),
                            dists.Normal(scale=np.sqrt(s2 + self.sigma_J**2)))
        return mix.logpdf(data[t + 1])

    def proposal(self, t, xp, data):
        y = data[t]; s2 = np.exp(xp); vJ = s2 + self.sigma_J**2; s = np.sqrt(s2)
        # proba a posteriori du saut  pi1 = P(J_t=1 | h_t, y_t)
        lw0 = np.log(1 - self.kappa) + dists.Normal(scale=s).logpdf(y)
        lw1 = np.log(self.kappa)     + dists.Normal(scale=np.sqrt(vJ)).logpdf(y)
        pi1 = 1.0 / (1.0 + np.exp(lw0 - lw1))
        ex = self.EXt(xp); sg = self.sigma; ph = self.phi
        # composante sans saut : eps = y e^{-h_t/2}
        c0 = dists.Normal(loc=ex + sg*ph*(y/s), scale=sg*np.sqrt(1 - ph**2))
        # composante avec saut : eps ~ N(upsilon1, sigma1^2)
        up1 = s * y / vJ; v1 = self.sigma_J**2 / vJ
        c1 = dists.Normal(loc=ex + sg*ph*up1, scale=sg*np.sqrt(ph**2 * v1 + 1 - ph**2))
        return dists.Mixture(np.column_stack([1 - pi1, pi1]), c0, c1)








