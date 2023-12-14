import sys
sys.path.insert(0, '../')
import bayesnewton.kernels as kernels
import numpy as np
from bayesnewton.utils import transpose
from jax import vmap
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor

np.random.seed(3)


def wiggly_time_series(x_):
    noise_var = 0.15  # true observation noise
    return (np.cos(0.04*x_+0.33*np.pi) * np.sin(0.2*x_) +
            np.sqrt(noise_var) * np.random.normal(0, 1, x_.shape) +
            0.0 * x_)  # 0.02 * x_)


print('generating some data ...')
np.random.seed(12345)
N = 10
x = np.linspace(-17, 147, num=N)
x = np.sort(x, axis=0)
y = wiggly_time_series(x)

x = x[:, None]

var_f = 1.0  # GP variance
len_f = 5.0  # GP lengthscale

kernel = kernels.Matern32(variance=var_f, lengthscale=len_f)


def prior_log_normaliser_gp():
    """ compute logZ using kernel """
    dim = 1  # TODO: implement multivariate case
    K = kernel.K(x, x)
    # (sign, logdet) = np.linalg.slogdet(K)
    C, low = cho_factor(K)
    logdet = 2 * jnp.sum(jnp.log(jnp.abs(jnp.diag(C))))
    logZ_prior = -0.5 * x.shape[0] * dim * np.log(2 * np.pi) - 0.5 * logdet
    return logZ_prior


def process_noise_covariance(A, Pinf):
    """ compute stationary noise cov, Q """
    Q = Pinf - A @ Pinf @ transpose(A)
    return Q


def diag(P):
    """ broadcastable version of jnp.diag """
    return vmap(jnp.diag)(P)


def prior_log_normaliser_markovgp():
    """ compute logZ using state space model """
    Pinf = kernel.stationary_covariance()
    dim = Pinf.shape[0]
    dt = jnp.diff(x[:, 0])
    As = vmap(kernel.state_transition)(dt)
    Qs = np.concatenate([Pinf[None], process_noise_covariance(As, Pinf)])
    Cs, low = cho_factor(Qs)
    logdet = 2 * jnp.sum(jnp.log(jnp.abs(diag(Cs))))
    # logZ_prior = -0.5 * x.shape[0] * dim * np.log(2 * np.pi) - 0.5 * logdet
    logZ_prior = -0.5 * x.shape[0] * np.log(2 * np.pi) - 0.5 * logdet
    return logZ_prior


logZ = prior_log_normaliser_gp()
print(logZ)

logZ_markov = prior_log_normaliser_markovgp()
print(logZ_markov)
