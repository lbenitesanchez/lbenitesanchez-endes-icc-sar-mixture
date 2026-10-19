"""Stable evaluation of the SAME collapsed beta power likelihood.

No precision bound or prior change. Gamma increments use an exact recurrence
below shape8 and an eight-term Stirling expansion above8. The first omitted
Stirling term is <8.0e-17 at8. Log1p's auxiliary small-ratio series has remainder
<=t**9/10 for0<=t<0.01. Values and analytic derivatives are audited separately.
"""
import numpy as np
from scipy.special import gammaln, digamma, expit, log_expit

STIRLING = ((1,1/12),(3,-1/360),(5,1/1260),(7,-1/1680),
            (9,1/1188),(11,-691/360360),(13,1/156),(15,-3617/122400))
LOG_SHAPE_SWITCH = float(np.log(8.0))


def gamma_increment_remainder(log_shape, increment):
    """R=logGamma(z+c)-logGamma(z)-c*log(z), dR/dlog(z)."""
    logz, c = np.broadcast_arrays(np.asarray(log_shape,float),np.asarray(increment,float))
    if np.any(c < 0): raise ValueError('Negative gamma increment')
    value = np.zeros(logz.shape); derivative = np.zeros(logz.shape)
    small = (c > 0) & (logz < LOG_SHAPE_SWITCH)
    z, k = np.exp(logz[small]), c[small]
    # Gamma(z)=Gamma(1+z)/z avoids underflow in Gamma of a tiny shape.
    value[small] = gammaln(z+k)-gammaln(1+z)+(1-k)*logz[small]
    derivative[small] = z*(digamma(z+k)-digamma(1+z))+(1-k)
    large = (c > 0) & ~small
    inverse, k = np.exp(-logz[large]), c[large]
    t = k*inverse; lt = np.log1p(t)
    h = np.empty_like(t); g = np.empty_like(t)
    series = t < .01
    h[series] = 0.; g[series] = 0.
    for m in range(8,0,-1):
        coefficient = (-1.)**m/(m+1)
        h[series] = t[series]*(h[series]+coefficient)
        g[series] = t[series]*(g[series]-m*coefficient)
    h[~series] = lt[~series]/t[~series]-1
    g[~series] = lt[~series]/t[~series]-1/(1+t[~series])
    r = k*h+(k-.5)*lt
    dr = k*g-(k-.5)*t/(1+t)
    for order, coefficient in STIRLING:
        e = np.expm1(-order*lt)
        term = coefficient*inverse**order
        r += term*e
        dr += term*(-order*e+(1+e)*order*t/(1+t))
    value[large] = r; derivative[large] = dr
    return value, derivative


def log_likelihood_and_gradient(eta, log_kappa, successes, failures):
    eta,k,f = np.broadcast_arrays(np.asarray(eta,float),np.asarray(successes,float),np.asarray(failures,float))
    lm, lb = log_expit(eta), log_expit(-eta)
    ra, da = gamma_increment_remainder(log_kappa+lm,k)
    rb, db = gamma_increment_remainder(log_kappa+lb,f)
    rc, dc = gamma_increment_remainder(np.broadcast_to(log_kappa,eta.shape),k+f)
    mu, complement = expit(eta), expit(-eta)
    ll = k*lm+f*lb+ra+rb-rc
    deta = k*complement-f*mu+da*complement-db*mu
    dlk = da+db-dc
    return ll, deta, dlk
