import jax.numpy as np
from jax import vmap
from jax.scipy.linalg import cho_factor, cho_solve
from .utils import diag, mvn_logpdf, solve, transpose, inv
from jax.lax import scan
import math

INV2PI = (2 * math.pi) ** -1


def gaussian_conditional(kernel, y, noise_cov, X, X_star=None):
    """
    Compute the GP posterior / predictive distribution using standard Gaussian identities
    :param kernel: an instantiation of the kernel class
    :param y: observations [N, 1]
    :param noise_cov: observation noise covariance [N, 1]
    :param X: training inputs [N, D]
    :param X_star: test inputs [N*, D]
    :return:
        mean: posterior mean [N, 1]
        covariance: posterior covariance [N, N]
    """
    Kff = kernel(X, X)
    if X_star is None:  # inference / learning
        Kfs = Kff
        Kss = Kff
    else:  # prediction
        Kfs = kernel(X, X_star)
        Kss = kernel(X_star, X_star)

    Ky = Kff + np.diag(np.squeeze(noise_cov))  # TODO: will break for multi-latents
    # ---- compute approximate posterior using standard Gaussian conditional formula ----
    Ly, low = cho_factor(Ky)
    Kfs_iKy = cho_solve((Ly, low), Kfs).T
    mean = Kfs_iKy @ diag(y)
    covariance = Kss - Kfs_iKy @ Kfs
    return mean, covariance


def sparse_gaussian_conditional(kernel, nat1lik, nat2lik, X, Z):
    """
    Compute q(u)
    :param kernel: an instantiation of the kernel class
    :param nat1lik: likelihood first natural parameter [N, 1]
    :param nat2lik: likelihood noise precision [N, 1]
    :param X: training inputs [N, D]
    :param Z: inducing inputs [N*, D]
    :return:
        mean: posterior mean [N, 1]
        covariance: posterior covariance [N, N]
    """
    Kuf = kernel(Z, X)
    Kuu = kernel(Z, Z)
    nat2prior = inv(Kuu)
    Wuf = solve(Kuu, Kuf)  # conditional mapping, Kuu^-1 Kuf

    nat1lik_fullrank = Wuf @ np.squeeze(nat1lik, axis=-1)  # TODO: will break for multi-latents
    nat2lik_fullrank = Wuf @ np.diag(np.squeeze(nat2lik)) @ transpose(Wuf)

    nat1post = nat1lik_fullrank  # prior nat1 is zero
    nat2post = nat2prior + nat2lik_fullrank

    covariance = inv(nat2post)
    mean = covariance @ nat1post
    return mean, covariance


def sparse_conditional_post_to_data(kernel, post_mean, post_cov, X, Z):
    """
    Compute int p(f|u) q(u) du
    :param kernel: an instantiation of the kernel class
    :param post_mean: posterior mean [M, 1]
    :param post_cov: posterior covariance [M, M]
    :param X: training inputs [N, D]
    :param Z: inducing inputs [N*, D]
    :return:
        mean: posterior mean [N, 1]
        covariance: posterior covariance [N, N]
    """
    Kff = kernel(X, X)
    Kuf = kernel(Z, X)
    Kuu = kernel(Z, Z)
    Wuf = solve(Kuu, Kuf)  # conditional mapping, Kuu^-1 Kuf
    Qff = transpose(Kuf) @ Wuf  # Kfu Kuu^-1 Kuf

    conditional_cov = Kff - Qff

    mean_f = transpose(Wuf) @ np.squeeze(post_mean, axis=-1)
    cov_f = conditional_cov + transpose(Wuf) @ post_cov @ Wuf

    return mean_f, cov_f


def process_noise_covariance(A, Pinf):
    Q = Pinf - A @ Pinf @ transpose(A)
    return Q


def _sequential_kf(As, Qs, H, ys, noise_covs, m0, P0, masks, return_predict=False):

    def body(carry, inputs):
        y, A, Q, obs_cov, mask = inputs
        m, P, ell = carry
        m_ = A @ m
        P_ = A @ P @ A.T + Q

        obs_mean = H @ m_
        HP = H @ P_
        S = HP @ H.T + obs_cov

        ell_n = mvn_logpdf(y, obs_mean, S, mask)
        ell = ell + ell_n

        K = solve(S, HP).T
        m = m_ + K @ (y - obs_mean)
        P = P_ - K @ HP
        if return_predict:
            return (m, P, ell), (m_, P_)
        else:
            return (m, P, ell), (m, P)

    (_, _, loglik), (fms, fPs) = scan(f=body,
                                      init=(m0, P0, 0.),
                                      xs=(ys, As, Qs, noise_covs, masks))
    return loglik, fms, fPs


def kalman_filter(dt, kernel, y, noise_cov, mask=None, use_sequential=True, return_predict=False):
    """
    Run the Kalman filter to get p(fₙ|y₁,...,yₙ).
    Assumes a heteroscedastic Gaussian observation model, i.e. var is vector valued
    :param dt: step sizes [N, 1]
    :param kernel: an instantiation of the kernel class, used to determine the state space model
    :param y: observations [N, D, 1]
    :param noise_cov: observation noise covariances [N, D, D]
    :param mask: boolean mask for the observations (to indicate missing data locations) [N, D, 1]
    :param use_sequential: flag to switch between parallel and sequential implementation of Kalman filter
    :param return_predict: flag whether to return predicted state, rather than updated state
    :return:
        ell: the log-marginal likelihood log p(y), for hyperparameter optimisation (learning) [scalar]
        means: intermediate filtering means [N, state_dim, 1]
        covs: intermediate filtering covariances [N, state_dim, state_dim]
    """
    if mask is None:
        mask = np.zeros_like(y, dtype=bool)
    Pinf = kernel.stationary_covariance()
    minf = np.zeros([Pinf.shape[0], 1])

    As = vmap(kernel.state_transition)(dt)
    Qs = vmap(process_noise_covariance, [0, None])(As, Pinf)
    H = kernel.measurement_model()

    if use_sequential:
        ell, means, covs = _sequential_kf(As, Qs, H, y, noise_cov, minf, Pinf, mask, return_predict=return_predict)
    else:
        raise NotImplementedError("Parallel KF not implemented yet")
    return ell, (means, covs)


def _sequential_rts(fms, fPs, As, Qs, H, return_full):

    def body(carry, inputs):
        fm, fP, A, Q = inputs
        sm, sP = carry

        pm = A @ fm
        AfP = A @ fP
        pP = AfP @ A.T + Q

        C = solve(pP, AfP).T

        sm = fm + C @ (sm - pm)
        sP = fP + C @ (sP - pP) @ C.T
        if return_full:
            return (sm, sP), (sm, sP, C)
        else:
            return (sm, sP), (H @ sm, H @ sP @ H.T, C)

    _, (sms, sPs, gains) = scan(f=body,
                                init=(fms[-1], fPs[-1]),
                                xs=(fms, fPs, As, Qs),
                                reverse=True)
    return sms, sPs, gains


def rauch_tung_striebel_smoother(dt, kernel, filter_mean, filter_cov, return_full=False, use_sequential=True):
    """
    Run the RTS smoother to get p(fₙ|y₁,...,y_N),
    :param dt: step sizes [N, 1]
    :param kernel: an instantiation of the kernel class, used to determine the state space model
    :param filter_mean: the intermediate distribution means computed during filtering [N, state_dim, 1]
    :param filter_cov: the intermediate distribution covariances computed during filtering [N, state_dim, state_dim]
    :param return_full: a flag determining whether to return the full state distribution or just the function(s)
    :param use_sequential: flag to switch between parallel and sequential implementation of smoother
    :return:
        smoothed_mean: the posterior marginal means [N, obs_dim]
        smoothed_var: the posterior marginal variances [N, obs_dim]
    """
    Pinf = kernel.stationary_covariance()

    As = vmap(kernel.state_transition)(dt)
    Qs = vmap(process_noise_covariance, [0, None])(As, Pinf)
    H = kernel.measurement_model()

    if use_sequential:
        means, covs, gains = _sequential_rts(filter_mean, filter_cov, As, Qs, H, return_full)
    else:
        raise NotImplementedError("Parallel RTS not implemented yet")
    return means, covs, gains


def kalman_filter_pairs(dt, kernel, y, noise_cov, use_sequential=True):
    """
    A Kalman filter over pairs of states, in which y is [2state_dim, 1] and noise_cov is [2state_dim, 2state_dim]
    :param dt: step sizes [N, 1]
    :param kernel: an instantiation of the kernel class, used to determine the state space model
    :param y: observations [N, 2state_dim, 1]
    :param noise_cov: observation noise covariances [N, 2state_dim, 2state_dim]
    :param use_sequential: flag to switch between parallel and sequential implementation of Kalman filter
    :return:
        ell: the log-marginal likelihood log p(y), for hyperparameter optimisation (learning) [scalar]
        means: marginal state filtering means [N, state_dim, 1]
        covs: marginal state filtering covariances [N, state_dim, state_dim]
    """
    Pinf = kernel.stationary_covariance()
    state_dim = Pinf.shape[0]
    minf = np.zeros([state_dim, 1])
    zeros = np.zeros([state_dim, state_dim])
    Pinfpair = np.block([[Pinf,  zeros],
                         [zeros, Pinf]])
    minfpair = np.block([[minf],
                         [minf]])

    As = vmap(kernel.state_transition)(dt)
    Qs = vmap(process_noise_covariance, [0, None])(As, Pinf)

    def construct_pair(A, Q):
        Apair = np.block([[zeros, np.eye(state_dim)],
                          [zeros, A]])
        Qpair = np.block([[zeros, zeros],
                          [zeros, Q]])
        return Apair, Qpair

    Apairs, Qpairs = vmap(construct_pair)(As, Qs)
    H = np.eye(2 * state_dim)
    masks = np.zeros_like(y, dtype=bool)

    if use_sequential:
        ell, means, covs = _sequential_kf(Apairs, Qpairs, H, y, noise_cov, minfpair, Pinfpair, masks)
    else:
        raise NotImplementedError("Parallel KF not implemented yet")
    return ell, (means[1:, :state_dim], covs[1:, :state_dim, :state_dim])
