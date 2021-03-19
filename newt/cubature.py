import objax
from jax import vmap
import jax.numpy as np
from jax.ops import index_add, index
from jax.scipy.linalg import cholesky, cho_factor
from .utils import solve, gaussian_first_derivative_wrt_mean, gaussian_second_derivative_wrt_mean
from numpy.polynomial.hermite import hermgauss
import itertools


class Cubature(objax.Module):

    def __call__(self, dim):
        raise NotImplementedError


class GaussHermite(Cubature):

    def __init__(self, num_cub_points=20):
        self.num_cub_points = num_cub_points

    def __call__(self, dim):
        return gauss_hermite(dim, self.num_cub_points)


class UnscentedThirdOrder(Cubature):

    def __call__(self, dim):
        return symmetric_cubature_third_order(dim)


class UnscentedFifthOrder(Cubature):

    def __call__(self, dim):
        return symmetric_cubature_fifth_order(dim)


class Unscented(UnscentedFifthOrder):
    pass


def mvhermgauss(H: int, D: int):
    """
    This function is adapted from GPflow: https://github.com/GPflow/GPflow

    Return the evaluation locations 'xn', and weights 'wn' for a multivariate
    Gauss-Hermite quadrature.

    The outputs can be used to approximate the following type of integral:
    int exp(-x)*f(x) dx ~ sum_i w[i,:]*f(x[i,:])

    :param H: Number of Gauss-Hermite evaluation points.
    :param D: Number of input dimensions. Needs to be known at call-time.
    :return: eval_locations 'x' (H**DxD), weights 'w' (H**D)
    """
    gh_x, gh_w = hermgauss(H)
    x = np.array(list(itertools.product(*(gh_x,) * D)))  # H**DxD
    w = np.prod(np.array(list(itertools.product(*(gh_w,) * D))), 1)  # H**D
    return x, w


def gauss_hermite(dim=1, num_quad_pts=20):
    """
    Return weights and sigma-points for Gauss-Hermite cubature
    """
    # sigma_pts, weights = hermgauss(num_quad_pts)  # Gauss-Hermite sigma points and weights
    sigma_pts, weights = mvhermgauss(num_quad_pts, dim)
    sigma_pts = np.sqrt(2) * sigma_pts.T
    weights = weights.T * np.pi ** (-0.5 * dim)  # scale weights by 1/√π
    return sigma_pts, weights


def symmetric_cubature_third_order(dim=1, kappa=None):
    """
    Return weights and sigma-points for the symmetric cubature rule of order 5, for
    dimension dim with parameter kappa (default 0).
    """
    if kappa is None:
        # kappa = 1 - dim
        kappa = 0  # CKF
    if (dim == 1) and (kappa == 0):
        weights = np.array([0., 0.5, 0.5])
        sigma_pts = np.array([0., 1., -1.])
        # sigma_pts = np.array([-1., 0., 1.])
        # weights = np.array([0.5, 0., 0.5])
        # u = 1
    elif (dim == 2) and (kappa == 0):
        weights = np.array([0., 0.25, 0.25, 0.25, 0.25])
        sigma_pts = np.block([[0., 1.4142,  0., -1.4142, 0.],
                              [0., 0., 1.4142, 0., -1.4142]])
        # u = 1.4142
    elif (dim == 3) and (kappa == 0):
        weights = np.array([0., 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667])
        sigma_pts = np.block([[0., 1.7321, 0.,  0., -1.7321, 0., 0.],
                              [0., 0., 1.7321, 0., 0., -1.7321, 0.],
                              [0., 0., 0., 1.7321, 0., 0., -1.7321]])
        # u = 1.7321
    else:
        # weights
        weights = np.zeros([1, 2 * dim + 1])
        weights = index_add(weights, index[0, 0], kappa / (dim + kappa))
        for j in range(1, 2 * dim + 1):
            wm = 1 / (2 * (dim + kappa))
            weights = index_add(weights, index[0, j], wm)
        # Sigma points
        sigma_pts = np.block([np.zeros([dim, 1]), np.eye(dim), - np.eye(dim)])
        sigma_pts = np.sqrt(dim + kappa) * sigma_pts
        # u = np.sqrt(n + kappa)
    return sigma_pts, weights  # , u


def symmetric_cubature_fifth_order(dim=1):
    """
    Return weights and sigma-points for the symmetric cubature rule of order 5
    TODO: implement general form
    """
    if dim == 1:
        weights = np.array([0.6667, 0.1667, 0.1667])
        sigma_pts = np.array([0., 1.7321, -1.7321])
    elif dim == 2:
        weights = np.array([0.4444, 0.1111, 0.1111, 0.1111, 0.1111, 0.0278, 0.0278, 0.0278, 0.0278])
        sigma_pts = np.block([[0., 1.7321, -1.7321, 0., 0., 1.7321, -1.7321, 1.7321, -1.7321],
                              [0., 0., 0., 1.7321, -1.7321, 1.7321, -1.7321, -1.7321, 1.7321]])
    elif dim == 3:
        weights = np.array([0.3333, 0.0556, 0.0556, 0.0556, 0.0556, 0.0556, 0.0556, 0.0278, 0.0278, 0.0278,
                            0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278])
        sigma_pts = np.block([[0., 1.7321, -1.7321, 0., 0., 0., 0., 1.7321, -1.7321, 1.7321, -1.7321, 1.7321,
                               -1.7321, 1.7321, -1.7321, 0., 0., 0., 0.],
                              [0., 0., 0., 1.7321, -1.7321, 0., 0., 1.7321, -1.7321, -1.7321, 1.7321, 0., 0., 0.,
                               0., 1.7321, -1.7321, 1.7321, -1.7321],
                              [0., 0., 0., 0., 0., 1.7321, -1.7321, 0., 0., 0., 0., 1.7321, -1.7321, -1.7321,
                               1.7321, 1.7321, -1.7321, -1.7321, 1.7321]])
    else:
        raise NotImplementedError
    return sigma_pts, weights


def variational_expectation_cubature(likelihood, y, post_mean, post_cov, cubature=None):
    """
    Computes the "variational expectation" via cubature, i.e. the
    expected log-likelihood, and its derivatives w.r.t. the posterior mean
        E[log p(yₙ|fₙ)] = ∫ log p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
    with EP power a.
    :param likelihood: the likelihood model
    :param y: observed data (yₙ) [scalar]
    :param post_mean: posterior mean (mₙ) [scalar]
    :param post_cov: posterior variance (vₙ) [scalar]
    :param cubature: the function to compute sigma points and weights to use during cubature
    :return:
        exp_log_lik: the expected log likelihood, E[log p(yₙ|fₙ)]  [scalar]
        dE_dm: derivative of E[log p(yₙ|fₙ)] w.r.t. mₙ  [scalar]
        d2E_dm2: second derivative of E[log p(yₙ|fₙ)] w.r.t. mₙ  [scalar]
    """
    if cubature is None:
        x, w = gauss_hermite(post_mean.shape[0], 20)  # Gauss-Hermite sigma points and weights
    else:
        x, w = cubature(post_mean.shape[0])
    # fsigᵢ=xᵢ√(vₙ) + mₙ: scale locations according to cavity dist.
    sigma_points = cholesky(post_cov) @ np.atleast_2d(x) + post_mean
    # pre-compute wᵢ log p(yₙ|xᵢ√(2vₙ) + mₙ)
    weighted_log_likelihood_eval = w * likelihood.evaluate_log_likelihood(y, sigma_points)
    # Compute expected log likelihood via cubature:
    # E[log p(yₙ|fₙ)] = ∫ log p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
    #                 ≈ ∑ᵢ wᵢ p(yₙ|fsigᵢ)
    exp_log_lik = np.sum(
        weighted_log_likelihood_eval
    )
    # Compute first derivative via cubature:
    # dE[log p(yₙ|fₙ)]/dmₙ = ∫ (fₙ-mₙ) vₙ⁻¹ log p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
    #                      ≈ ∑ᵢ wᵢ (fₙ-mₙ) vₙ⁻¹ log p(yₙ|fsigᵢ)
    invv = np.diag(post_cov)[:, None] ** -1
    dE_dm = np.sum(
        invv * (sigma_points - post_mean)
        * weighted_log_likelihood_eval, axis=-1
    )[:, None]
    # Compute second derivative via cubature (deriv. w.r.t. var = 0.5 * 2nd deriv. w.r.t. mean):
    # dE[log p(yₙ|fₙ)]/dvₙ = ∫ [(fₙ-mₙ)² vₙ⁻² - vₙ⁻¹]/2 log p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
    #                        ≈ ∑ᵢ wᵢ [(fₙ-mₙ)² vₙ⁻² - vₙ⁻¹]/2 log p(yₙ|fsigᵢ)
    dE_dv = np.sum(
        (0.5 * (invv ** 2 * (sigma_points - post_mean) ** 2) - 0.5 * invv)
        * weighted_log_likelihood_eval, axis=-1
    )
    dE_dv = np.diag(dE_dv)
    d2E_dm2 = 2 * dE_dv
    return exp_log_lik, dE_dm, d2E_dm2


def log_density_cubature(likelihood, y, mean, cov, cubature=None):
    """
    logZₙ = log ∫ p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
    :param likelihood: the likelihood model
    :param y: observed data (yₙ) [scalar]
    :param mean: cavity mean (mₙ) [scalar]
    :param cov: cavity covariance (cₙ) [scalar]
    :param cubature: the function to compute sigma points and weights to use during cubature
    :return:
        lZ: the log density, logZₙ  [scalar]
    """
    if cubature is None:
        x, w = gauss_hermite(mean.shape[0], 20)  # Gauss-Hermite sigma points and weights
    else:
        x, w = cubature(mean.shape[0])
    cav_cho, low = cho_factor(cov)
    # fsigᵢ=xᵢ√cₙ + mₙ: scale locations according to cavity dist.
    sigma_points = cav_cho @ np.atleast_2d(x) + mean
    # pre-compute wᵢ p(yₙ|xᵢ√(2vₙ) + mₙ)
    weighted_likelihood_eval = w * likelihood.evaluate_likelihood(y, sigma_points)
    # Compute partition function via cubature:
    # Zₙ = ∫ p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ ≈ ∑ᵢ wᵢ p(yₙ|fsigᵢ)
    Z = np.sum(
        weighted_likelihood_eval, axis=-1
    )
    lZ = np.log(np.maximum(Z, 1e-8))
    return lZ


def moment_match_cubature(likelihood, y, cav_mean, cav_cov, power=1.0, cubature=None):
    """
    TODO: N.B. THIS VERSION ALLOWS MULTI-DIMENSIONAL MOMENT MATCHING, BUT CAN BE UNSTABLE
    Perform moment matching via cubature.
    Moment matching involves computing the log partition function, logZₙ, and its derivatives w.r.t. the cavity mean
        logZₙ = log ∫ pᵃ(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
    with EP power a.
    :param likelihood: the likelihood model
    :param y: observed data (yₙ) [scalar]
    :param cav_mean: cavity mean (mₙ) [scalar]
    :param cav_cov: cavity covariance (cₙ) [scalar]
    :param power: EP power / fraction (a) [scalar]
    :param cubature: the function to compute sigma points and weights to use during cubature
    :return:
        lZ: the log partition function, logZₙ  [scalar]
        dlZ: first derivative of logZₙ w.r.t. mₙ (if derivatives=True)  [scalar]
        d2lZ: second derivative of logZₙ w.r.t. mₙ (if derivatives=True)  [scalar]
    """
    if cubature is None:
        x, w = gauss_hermite(cav_mean.shape[0], 20)  # Gauss-Hermite sigma points and weights
    else:
        x, w = cubature(cav_mean.shape[0])
    cav_cho, low = cho_factor(cav_cov)
    # fsigᵢ=xᵢ√cₙ + mₙ: scale locations according to cavity dist.
    sigma_points = cav_cho @ np.atleast_2d(x) + cav_mean
    # pre-compute wᵢ pᵃ(yₙ|xᵢ√(2vₙ) + mₙ)
    weighted_likelihood_eval = w * likelihood.evaluate_likelihood(y, sigma_points) ** power

    # Compute partition function via cubature:
    # Zₙ = ∫ pᵃ(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
    #    ≈ ∑ᵢ wᵢ pᵃ(yₙ|fsigᵢ)
    Z = np.sum(
        weighted_likelihood_eval, axis=-1
    )
    lZ = np.log(np.maximum(Z, 1e-8))
    Zinv = 1.0 / np.maximum(Z, 1e-8)

    # Compute derivative of partition function via cubature:
    # dZₙ/dmₙ = ∫ (fₙ-mₙ) vₙ⁻¹ pᵃ(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
    #         ≈ ∑ᵢ wᵢ (fₙ-mₙ) vₙ⁻¹ pᵃ(yₙ|fsigᵢ)
    d1 = vmap(
        gaussian_first_derivative_wrt_mean, (1, None, None, 1)
    )(sigma_points[..., None], cav_mean, cav_cov, weighted_likelihood_eval)
    dZ = np.sum(d1, axis=0)
    # dlogZₙ/dmₙ = (dZₙ/dmₙ) / Zₙ
    dlZ = Zinv * dZ

    # Compute second derivative of partition function via cubature:
    # d²Zₙ/dmₙ² = ∫ [(fₙ-mₙ)² vₙ⁻² - vₙ⁻¹] pᵃ(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
    #           ≈ ∑ᵢ wᵢ [(fₙ-mₙ)² vₙ⁻² - vₙ⁻¹] pᵃ(yₙ|fsigᵢ)
    d2 = vmap(
        gaussian_second_derivative_wrt_mean, (1, None, None, 1)
    )(sigma_points[..., None], cav_mean, cav_cov, weighted_likelihood_eval)
    d2Z = np.sum(d2, axis=0)

    # d²logZₙ/dmₙ² = d[(dZₙ/dmₙ) / Zₙ]/dmₙ
    #              = (d²Zₙ/dmₙ² * Zₙ - (dZₙ/dmₙ)²) / Zₙ²
    #              = d²Zₙ/dmₙ² / Zₙ - (dlogZₙ/dmₙ)²
    d2lZ = -dlZ @ dlZ.T + Zinv * d2Z
    return lZ, dlZ, d2lZ


def statistical_linear_regression_cubature(likelihood, mean, cov, cubature=None):
    """
    Perform statistical linear regression (SLR) using cubature.
    We aim to find a likelihood approximation p(yₙ|fₙ) ≈ 𝓝(yₙ|Afₙ+b,Ω+Var[yₙ|fₙ]).
    TODO: this currently assumes an additive noise model (ok for our current applications), make more general
    """
    if cubature is None:
        x, w = gauss_hermite(mean.shape[0], 20)  # Gauss-Hermite sigma points and weights
    else:
        x, w = cubature(mean.shape[0])
    # fsigᵢ=xᵢ√(vₙ) + mₙ: scale locations according to cavity dist.
    sigma_points = cholesky(cov) @ np.atleast_2d(x) + mean
    lik_expectation, lik_covariance = likelihood.conditional_moments(sigma_points)
    # Compute zₙ via cubature:
    # zₙ = ∫ E[yₙ|fₙ] 𝓝(fₙ|mₙ,vₙ) dfₙ
    #    ≈ ∑ᵢ wᵢ E[yₙ|fsigᵢ]
    mu = np.sum(
        w * lik_expectation, axis=-1
    )[:, None]
    # Compute variance S via cubature:
    # S = ∫ [(E[yₙ|fₙ]-zₙ) (E[yₙ|fₙ]-zₙ)' + Cov[yₙ|fₙ]] 𝓝(fₙ|mₙ,vₙ) dfₙ
    #   ≈ ∑ᵢ wᵢ [(E[yₙ|fsigᵢ]-zₙ) (E[yₙ|fsigᵢ]-zₙ)' + Cov[yₙ|fₙ]]
    # TODO: allow for multi-dim cubature
    S = np.sum(
        w * ((lik_expectation - mu) * (lik_expectation - mu) + lik_covariance), axis=-1
    )[:, None]
    # Compute cross covariance C via cubature:
    # C = ∫ (fₙ-mₙ) (E[yₙ|fₙ]-zₙ)' 𝓝(fₙ|mₙ,vₙ) dfₙ
    #   ≈ ∑ᵢ wᵢ (fsigᵢ -mₙ) (E[yₙ|fsigᵢ]-zₙ)'
    C = np.sum(
        w * (sigma_points - mean) * (lik_expectation - mu), axis=-1
    )[:, None]
    # Compute derivative of z via cubature:
    # d_mu = ∫ E[yₙ|fₙ] vₙ⁻¹ (fₙ-mₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
    #      ≈ ∑ᵢ wᵢ E[yₙ|fsigᵢ] vₙ⁻¹ (fsigᵢ-mₙ)
    d_mu = np.sum(
        w * lik_expectation * (solve(cov, sigma_points - mean)), axis=-1
    )[None, :]
    return mu, S, C, d_mu


def predict_cubature(likelihood, mean_f, var_f, cubature=None):
    """
    predict in data space given predictive mean and var of the latent function
    """
    if cubature is None:
        x, w = gauss_hermite(mean_f.shape[0], 20)  # Gauss-Hermite sigma points and weights
    else:
        x, w = cubature(mean_f.shape[0])
    chol_f, low = cho_factor(var_f)
    # fsigᵢ=xᵢ√cₙ + mₙ: scale locations according to latent dist.
    sigma_points = chol_f @ np.atleast_2d(x) + mean_f
    # Compute moments via cubature:
    # E[y] = ∫ E[yₙ|fₙ] 𝓝(fₙ|mₙ,vₙ) dfₙ
    #      ≈ ∑ᵢ wᵢ E[yₙ|fₙ]
    # E[y^2] = ∫ (Cov[yₙ|fₙ] + E[yₙ|fₙ]^2) 𝓝(fₙ|mₙ,vₙ) dfₙ
    #        ≈ ∑ᵢ wᵢ (Cov[yₙ|fₙ] + E[yₙ|fₙ]^2)
    conditional_expectation, conditional_covariance = likelihood.conditional_moments(sigma_points)
    expected_y = np.sum(w * conditional_expectation, axis=-1)
    expected_y_squared = np.sum(w * (conditional_covariance + conditional_expectation ** 2), axis=-1)
    # Cov[y] = E[y^2] - E[y]^2
    covariance_y = expected_y_squared - expected_y ** 2
    return expected_y, covariance_y
