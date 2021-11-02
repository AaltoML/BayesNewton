import objax
import jax.numpy as np
from jax import grad, jacrev
from jax.scipy.linalg import cholesky
from bayesnewton.likelihoods import Likelihood
from bayesnewton.utils import softplus, softplus_inv, sigmoid, sigmoid_diff
from bayesnewton.cubature import expected_conditional_mean_cubature, gauss_hermite


class Positive(Likelihood):
    """
    """
    def __init__(self, variance=0.1):
        """
        param hyp: observation noise
        """
        self.transformed_variance = objax.TrainVar(np.array(softplus_inv(variance)))
        super().__init__()
        self.name = 'Positive'
        self.link_fn = pos_map
        self.dlink_fn = dpos_map  # derivative of the link function
        self.d2link_fn = d2pos_map   # 2nd derivative of the link function

    @property
    def variance(self):
        return softplus(self.transformed_variance.value)

    def evaluate_likelihood(self, y, f):
        """
        Evaluate the likelihood
        """
        mu, var = self.conditional_moments(f)
        return (2 * np.pi * var) ** -0.5 * np.exp(-0.5 * (y - mu) ** 2 / var)

    def evaluate_log_likelihood(self, y, f):
        """
        Evaluate the log-likelihood
        """
        mu, var = self.conditional_moments(f)
        return np.squeeze(-0.5 * np.log(2 * np.pi * var) - 0.5 * (y - mu) ** 2 / var)

    def conditional_moments(self, f):
        """
        """
        return self.link_fn(f), np.array([[self.variance]])

    def log_likelihood_gradients(self, y, f):
        log_lik, J, H = self.log_likelihood_gradients_(y, f)
        return log_lik, J, H

    def expected_conditional_mean(self, mean, cov, cubature=None):
        return expected_conditional_mean_cubature(self, mean, cov, cubature)

    def expected_conditional_mean_dm(self, mean, cov, cubature=None):
        """
        """
        dmu_dm, _ = grad(self.expected_conditional_mean, argnums=0, has_aux=True)(mean, cov, cubature)
        return np.squeeze(dmu_dm)

    def expected_conditional_mean_dm2(self, mean, cov, cubature=None):
        """
        """
        d2mu_dm2 = jacrev(self.expected_conditional_mean_dm, argnums=0)(mean, cov, cubature)
        return d2mu_dm2

    def statistical_linear_regression(self, mean, cov, cubature=None):
        mu, omega = self.expected_conditional_mean(mean, cov, cubature)
        dmu_dm = self.expected_conditional_mean_dm(mean, cov, cubature)
        d2mu_dm2 = self.expected_conditional_mean_dm2(mean, cov, cubature)
        return mu.reshape(-1, 1), omega, dmu_dm[None, None], d2mu_dm2[None]


def _gaussian_expected_log_lik(y, post_mean, post_cov, var):
    post_mean = post_mean.reshape(-1, 1)
    post_cov = post_cov.reshape(-1, 1)
    y = y.reshape(-1, 1)
    var = var.reshape(-1, 1)
    # version which computes individual parts and outputs vector
    exp_log_lik = (
        -0.5 * np.log(2 * np.pi)
        - 0.5 * np.log(var)
        - 0.5 * ((y - post_mean) ** 2 + post_cov) / var
    )
    return exp_log_lik


def _gaussian_expected_log_lik_positive(y, post_mean, post_cov, var):
    post_mean = post_mean.reshape(-1, 1)
    post_cov = post_cov.reshape(-1, 1)
    y = y.reshape(-1, 1)
    var = var.reshape(-1, 1)

    x, w = gauss_hermite(post_mean.shape[0], 20)
    sigma_points = cholesky(post_cov) @ np.atleast_2d(x) + post_mean
    conditional_expectation = np.sum(w * pos_map(sigma_points))

    square_error_cubature = ((y - conditional_expectation) ** 2 + post_cov) / var

    # version which computes individual parts and outputs vector
    exp_log_lik = (
        -0.5 * np.log(2 * np.pi)
        - 0.5 * np.log(var)
        - 0.5 * square_error_cubature
    )
    return exp_log_lik


def _gaussian_expected_log_lik_positive_attempt(y, post_mean, post_cov, var):
    post_mean = post_mean.reshape(-1, 1)
    post_cov = post_cov.reshape(-1, 1)
    y = y.reshape(-1, 1)
    var = var.reshape(-1, 1)

    x, w = gauss_hermite(post_mean.shape[0], 20)
    sigma_points = cholesky(post_cov) @ np.atleast_2d(x) + post_mean
    conditional_expectation = np.sum(w * pos_map(sigma_points))

    def cov_term(mean):
        return (conditional_expectation - pos_map(mean)) ** 2

    final_term = np.sum(w * cov_term(sigma_points))

    square_error_cubature = ((y - conditional_expectation) ** 2 + final_term) / var

    # version which computes individual parts and outputs vector
    exp_log_lik = (
        -0.5 * np.log(2 * np.pi)
        - 0.5 * np.log(var)
        - 0.5 * square_error_cubature
    )
    return exp_log_lik


def var_exp_positive(y, post_mean, post_cov, var):
    post_mean = post_mean.reshape(-1, 1)
    post_cov = post_cov.reshape(-1, 1)
    y = y.reshape(-1, 1)
    var = var.reshape(-1, 1)

    def square_error(mean):
        return (y - pos_map(mean)) ** 2 * var ** -1

    x, w = gauss_hermite(post_mean.shape[0], 20)
    sigma_points = cholesky(post_cov) @ np.atleast_2d(x) + post_mean

    square_error_cubature = np.sum(w * square_error(sigma_points))

    # version which computes individual parts and outputs vector
    exp_log_lik = (
        -0.5 * np.log(2 * np.pi)
        - 0.5 * np.log(var)
        - 0.5 * square_error_cubature
    )
    return exp_log_lik


def pos_map(q):
    return softplus(q)


def dpos_map(q):
    return sigmoid(q)


def d2pos_map(q):
    return sigmoid_diff(q)


y, m, v, obs_var = np.array(0.2), np.array(0.1), np.array(0.1), np.array(0.5)

# var_exp = _gaussian_expected_log_lik(y, m, v, obs_var)
var_exp_pos = _gaussian_expected_log_lik_positive(y, m, v, obs_var)
var_exp_pos_attempt = _gaussian_expected_log_lik_positive_attempt(y, m, v, obs_var)
var_exp_pos_true = var_exp_positive(y, m, v, obs_var)

# lik = Positive(variance=obs_var)
# var_exp_pos_true_ = lik.variational_expectation_(y[None], m[None, None], v[None, None])

# print(var_exp)
print(var_exp_pos)
print(var_exp_pos_attempt)
print(var_exp_pos_true)
# print(var_exp_pos_true_)
