import jax.numpy as np
import numpy as nnp
from jax import vmap
from jax.scipy.linalg import cholesky, cho_factor, cho_solve
from jax.scipy.special import gammaln
from jax.lax import scan
# from matplotlib._png import read_png
import math

LOG2PI = math.log(2 * math.pi)
INV2PI = (2 * math.pi) ** -1


def solve(P, Q):
    """
    Compute P^-1 Q, where P is a PSD matrix, using the Cholesky factorisation
    """
    L = cho_factor(P, lower=True)
    return cho_solve(L, Q)


def inv(P):
    """
    Compute the inverse of a PSD matrix using the Cholesky factorisation
    """
    L = cho_factor(P, lower=True)
    return cho_solve(L, np.eye(P.shape[-1]))


def inv_vmap(P):
    """
    Compute the inverse of a PSD matrix using the Cholesky factorisation
    """
    L = cho_factor(P, lower=True)
    return cho_solve(L, np.tile(np.eye(P.shape[-1]), [P.shape[0], 1, 1]))


def diag(P):
    """
    a broadcastable version of np.diag, for when P is size [N, D, D]
    """
    return np.diagonal(P, axis1=1, axis2=2)


@vmap
def vmap_diag(P):
    return np.diag(P)


def transpose(P):
    return np.swapaxes(P, -1, -2)


def softplus(x_):
    return np.log(1. + np.exp(x_))
    # return np.log(1. + np.exp(-np.abs(x_))) + np.maximum(x_, 1e-24)  # safer version (but derivatve can have issues)


def sigmoid(x_):
    return np.exp(x_) / (np.exp(x_) + 1.)


def sigmoid_diff(x_):
    return sigmoid(x_) * (1. - sigmoid(x_))


def softplus_inv(x_):
    """
    Inverse of the softplus positiviy mapping, used for transforming parameters.
    """
    if x_ is None:
        return x_
    else:
        return np.log(np.exp(x_) - 1)
        # return np.log(1. - np.exp(-np.abs(x_))) + np.maximum(x_, 1e-24)  # safer version


def ensure_positive_precision(K):
    """
    Check whether matrix K has positive diagonal elements.
    If not, then replace the negative elements with default value 0.01
    """
    # K_diag = diag(diag(K))
    K_diag = vmap_diag(diag(K))
    K = np.where(np.any(diag(K) < 0), np.where(K_diag < 0, 1e-2, K_diag), K)
    return K


def ensure_diagonal_positive_precision(K):
    """
    Return a diagonal matrix with all positive values.
    """
    # K_diag = diag(K)[..., None] * np.eye(K.shape[1])
    K_diag = vmap_diag(diag(K))
    K = np.where(K_diag < 0, 1e-2, K_diag)
    return K


def predict_from_state(x_test, ind, x, post_mean, post_cov, gain, kernel):
    """
    wrapper function to vectorise predict_at_t_()
    """
    predict_from_state_func = vmap(
        predict_from_state_, (0, 0, None, None, None, None, None)
    )
    return predict_from_state_func(x_test, ind, x, post_mean, post_cov, gain, kernel)


def predict_from_state_(x_test, ind, x, post_mean, post_cov, gain, kernel):
    """
    predict the state distribution at time t by projecting from the neighbouring inducing states
    """
    P, T = compute_conditional_statistics(x_test, x, kernel, ind)
    # joint posterior (i.e. smoothed) mean and covariance of the states [u_, u+] at time t:
    mean_joint = np.block([[post_mean[ind]],
                           [post_mean[ind + 1]]])
    cross_cov = gain[ind] @ post_cov[ind + 1]
    cov_joint = np.block([[post_cov[ind], cross_cov],
                          [cross_cov.T, post_cov[ind + 1]]])
    return P @ mean_joint, P @ cov_joint @ P.T + T


def temporal_conditional(X, X_test, mean, cov, gain, kernel):
    """
    predict from time X to time X_test give state mean and covariance at X
    """
    Pinf = kernel.stationary_covariance()[None, ...]
    minf = np.zeros([1, Pinf.shape[1], 1])
    mean_aug = np.concatenate([minf, mean, minf])
    cov_aug = np.concatenate([Pinf, cov, Pinf])
    gain = np.concatenate([np.zeros_like(gain[:1]), gain])

    # figure out which two training states each test point is located between
    ind_test = np.searchsorted(X.reshape(-1, ), X_test.reshape(-1, )) - 1

    # project from training states to test locations
    test_mean, test_cov = predict_from_state(X_test, ind_test, X, mean_aug, cov_aug, gain, kernel)

    return test_mean, test_cov


def predict_from_state_infinite_horizon(x_test, ind, x, post_mean, kernel):
    """
    wrapper function to vectorise predict_at_t_()
    """
    predict_from_state_func = vmap(
        predict_from_state_infinite_horizon_, (0, 0, None, None, None)
    )
    return predict_from_state_func(x_test, ind, x, post_mean, kernel)


def predict_from_state_infinite_horizon_(x_test, ind, x, post_mean, kernel):
    """
    predict the state distribution at time t by projecting from the neighbouring inducing states
    """
    P, T = compute_conditional_statistics(x_test, x, kernel, ind)
    # joint posterior (i.e. smoothed) mean and covariance of the states [u_, u+] at time t:
    mean_joint = np.block([[post_mean[ind]],
                           [post_mean[ind + 1]]])
    return P @ mean_joint


def temporal_conditional_infinite_horizon(X, X_test, mean, cov, gain, kernel):
    """
    predict from time X to time X_test give state mean and covariance at X
    """
    Pinf = kernel.stationary_covariance()[None, ...]
    minf = np.zeros([1, Pinf.shape[1], 1])
    mean_aug = np.concatenate([minf, mean, minf])

    # figure out which two training states each test point is located between
    ind_test = np.searchsorted(X.reshape(-1, ), X_test.reshape(-1, )) - 1

    # project from training states to test locations
    test_mean = predict_from_state_infinite_horizon(X_test, ind_test, X, mean_aug, kernel)

    return test_mean, np.tile(cov[0], [test_mean.shape[0], 1, 1])


def compute_conditional_statistics(x_test, x, kernel, ind):
    """
    This version uses cho_factor and cho_solve - much more efficient when using JAX

    Predicts marginal states at new time points. (new time points should be sorted)
    Calculates the conditional density:
             p(xₙ|u₋, u₊) = 𝓝(Pₙ @ [u₋, u₊], Tₙ)

    :param x_test: time points to generate observations for [N]
    :param x: inducing state input locations [M]
    :param kernel: prior object providing access to state transition functions
    :param ind: an array containing the index of the inducing state to the left of every input [N]
    :return: parameters for the conditional mean and covariance
            P: [N, D, 2*D]
            T: [N, D, D]
    """
    dt_fwd = x_test[..., 0] - x[ind, 0]
    dt_back = x[ind + 1, 0] - x_test[..., 0]
    A_fwd = kernel.state_transition(dt_fwd)
    A_back = kernel.state_transition(dt_back)
    Pinf = kernel.stationary_covariance()
    Q_fwd = Pinf - A_fwd @ Pinf @ A_fwd.T
    Q_back = Pinf - A_back @ Pinf @ A_back.T
    A_back_Q_fwd = A_back @ Q_fwd
    Q_mp = Q_back + A_back @ A_back_Q_fwd.T

    jitter = 1e-8 * np.eye(Q_mp.shape[0])
    chol_Q_mp = cho_factor(Q_mp + jitter, lower=True)
    Q_mp_inv_A_back = cho_solve(chol_Q_mp, A_back)  # V = Q₋₊⁻¹ Aₜ₊

    # The conditional_covariance T = Q₋ₜ - Q₋ₜAₜ₊ᵀQ₋₊⁻¹Aₜ₊Q₋ₜ == Q₋ₜ - Q₋ₜᵀAₜ₊ᵀL⁻ᵀL⁻¹Aₜ₊Q₋ₜ
    T = Q_fwd - A_back_Q_fwd.T @ Q_mp_inv_A_back @ Q_fwd
    # W = Q₋ₜAₜ₊ᵀQ₋₊⁻¹
    W = Q_fwd @ Q_mp_inv_A_back.T
    P = np.concatenate([A_fwd - W @ A_back @ A_fwd, W], axis=-1)
    return P, T


def sum_natural_params_by_group(carry, inputs):
    ind_m, nat1_m, nat2_m = inputs
    nat1s, nat2s, count = carry
    nat1s = nat1s.at[ind_m].add(nat1_m)
    nat2s = nat2s.at[ind_m].add(nat2_m)
    count = count.at[ind_m].add(1.0)
    return (nat1s, nat2s, count), 0.


def count_indices(carry, inputs):
    ind_m = inputs
    count = carry
    count = count.at[ind_m].add(1.0)
    return count, 0.


def input_admin(t, y, r):
    """
    Order the inputs.
    :param t: training inputs [N, 1]
    :param y: observations at the training inputs [N, 1]
    :param r: training spatial inputs
    :return:
        t_train: training inputs [N, 1]
        y_train: training observations [N, R]
        r_train: training spatial inputs [N, R]
        dt_train: training step sizes, Δtₙ = tₙ - tₙ₋₁ [N, 1]
    """
    assert t.shape[0] == y.shape[0]
    if t.ndim < 2:
        t = nnp.expand_dims(t, 1)  # make 2-D
    if y.ndim < 2:
        y = nnp.expand_dims(y, 1)  # make 2-D
    if r is None:
        if t.shape[1] > 1:
            r = t[:, 1:]
            t = t[:, :1]
        else:
            r = nnp.nan * t  # np.empty((1,) + x.shape[1:]) * np.nan
    if r.ndim < 2:
        r = nnp.expand_dims(r, 1)  # make 2-D
    ind = nnp.argsort(t[:, 0], axis=0)
    t_train = t[ind, ...]
    y_train = y[ind, ...]
    r_train = r[ind, ...]
    dt_train = nnp.concatenate([np.array([0.0]), nnp.diff(t_train[:, 0])])
    return (np.array(t_train, dtype=np.float64), np.array(y_train, dtype=np.float64),
            np.array(r_train, dtype=np.float64), np.array(dt_train, dtype=np.float64))


def create_spatiotemporal_grid(X, Y):
    """
    create a grid of data sized [T, R1, R2]
    note that this function removes full duplicates (i.e. where all dimensions match)
    TODO: generalise to >5D
    """
    if Y.ndim < 2:
        Y = Y[:, None]
    num_spatial_dims = X.shape[1] - 1
    if num_spatial_dims == 4:
        sort_ind = nnp.lexsort((X[:, 4], X[:, 3], X[:, 2], X[:, 1], X[:, 0]))  # sort by 0, 1, 2, 4
    elif num_spatial_dims == 3:
        sort_ind = nnp.lexsort((X[:, 3], X[:, 2], X[:, 1], X[:, 0]))  # sort by 0, 1, 2, 3
    elif num_spatial_dims == 2:
        sort_ind = nnp.lexsort((X[:, 2], X[:, 1], X[:, 0]))  # sort by 0, 1, 2
    elif num_spatial_dims == 1:
        sort_ind = nnp.lexsort((X[:, 1], X[:, 0]))  # sort by 0, 1
    else:
        raise NotImplementedError
    X = X[sort_ind]
    Y = Y[sort_ind]
    unique_time = np.unique(X[:, 0])
    unique_space = nnp.unique(X[:, 1:], axis=0)
    N_t = unique_time.shape[0]
    N_r = unique_space.shape[0]
    if num_spatial_dims == 4:
        R = np.tile(unique_space, [N_t, 1, 1, 1, 1])
    elif num_spatial_dims == 3:
        R = np.tile(unique_space, [N_t, 1, 1, 1])
    elif num_spatial_dims == 2:
        R = np.tile(unique_space, [N_t, 1, 1])
    elif num_spatial_dims == 1:
        R = np.tile(unique_space, [N_t, 1])
    else:
        raise NotImplementedError
    R_flat = R.reshape(-1, num_spatial_dims)
    Y_dummy = np.nan * np.zeros([N_t * N_r, 1])
    time_duplicate = np.tile(unique_time, [N_r, 1]).T.flatten()
    X_dummy = np.block([time_duplicate[:, None], R_flat])
    X_all = np.vstack([X, X_dummy])
    Y_all = np.vstack([Y, Y_dummy])
    X_unique, ind = nnp.unique(X_all, axis=0, return_index=True)
    Y_unique = Y_all[ind]
    grid_shape = (unique_time.shape[0], ) + unique_space.shape
    R_grid = X_unique[:, 1:].reshape(grid_shape)
    Y_grid = Y_unique.reshape(grid_shape[:-1] + (1, ))
    return unique_time[:, None], R_grid, Y_grid


def discretegrid(xy, w, nt):
    """
    Convert spatial observations to a discrete intensity grid
    :param xy: observed spatial locations as a two-column vector
    :param w: observation window, i.e. discrete grid to be mapped to, [xmin xmax ymin ymax]
    :param nt: two-element vector defining number of bins in both directions
    """
    # Make grid
    x = nnp.linspace(w[0], w[1], nt[0] + 1)
    y = nnp.linspace(w[2], w[3], nt[1] + 1)
    X, Y = nnp.meshgrid(x, y)

    # Count points
    N = nnp.zeros([nt[1], nt[0]])
    for i in range(nt[0]):
        for j in range(nt[1]):
            ind = (xy[:, 0] >= x[i]) & (xy[:, 0] < x[i + 1]) & (xy[:, 1] >= y[j]) & (xy[:, 1] < y[j + 1])
            N[j, i] = nnp.sum(ind)
    return X[:-1, :-1].T, Y[:-1, :-1].T, N.T


def gaussian_log_expected_lik(y, post_mean, post_cov, obs_var):
    """
    Calculates the log partition function:
        logZₙ = log ∫ 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ = E[𝓝(yₙ|fₙ,σ²)]
    :param y: data / observation (yₙ)
    :param post_mean: posterior mean (mₙ)
    :param post_cov: posterior variance (vₙ)
    :param obs_var: variance, σ², of the Gaussian observation model p(yₙ|fₙ)=𝓝(yₙ|fₙ,σ²)
    :return:
        lZ: the log partition function, logZₙ [scalar]
    """
    post_mean = post_mean.reshape(-1, 1)
    post_var = np.diag(post_cov).reshape(-1, 1)
    y = y.reshape(-1, 1)
    obs_var = np.diag(obs_var).reshape(-1, 1)

    var = obs_var + post_var
    prec = 1 / var
    # version which computes sum and outputs scalar
    # lZ = (
    #     -0.5 * y.shape[-2] * np.log(2 * np.pi)
    #     - 0.5 * np.sum((y - post_mean) * prec * (y - post_mean))
    #     - 0.5 * np.sum(np.log(np.maximum(var, 1e-10)))
    # )
    # version which computes individual parts and outputs vector
    lZ = (
        -0.5 * np.log(2 * np.pi)
        - 0.5 * (y - post_mean) * prec * (y - post_mean)
        - 0.5 * np.log(np.maximum(var, 1e-10))
    )
    return lZ


def log_chol_matrix_det(chol):
    val = np.square(np.diag(chol))
    return np.sum(np.log(val))


def mvn_logpdf(x, mean, cov, mask=None):
    """
    evaluate a multivariate Gaussian (log) pdf
    """
    x = x.reshape(-1, 1)
    mean = mean.reshape(-1, 1)
    if mask is not None:
        # build a mask for computing the log likelihood of a partially observed multivariate Gaussian
        maskv = mask.reshape(-1, 1)
        x = np.where(maskv, 0., x)
        mean = np.where(maskv, 0., mean)
        cov_masked = np.where(maskv + maskv.T, 0., cov)  # ensure masked entries are independent
        cov = np.where(np.diag(mask), INV2PI, cov_masked)  # ensure masked entries return log like of 0

    n = mean.shape[0]
    cho, low = cho_factor(cov, lower=True)
    log_det = 2 * np.sum(np.log(np.abs(np.diag(cho))))
    diff = x - mean
    scaled_diff = cho_solve((cho, low), diff)
    distance = diff.T @ scaled_diff
    return np.squeeze(-0.5 * (distance + n * LOG2PI + log_det))


@vmap
def mvn_logpdf_(*args, **kwargs):
    return mvn_logpdf(*args, **kwargs)


def mvst_logpdf(x, mean, scale, df, mask=None):
    """
    evaluate a multivariate Student's t (log) pdf
    """
    x = x.reshape(-1, 1)
    mean = mean.reshape(-1, 1)
    dim = scale.shape[0]
    if mask is not None:
        maskv = mask.reshape(-1, 1)
        # build a mask for computing the log likelihood of a partially observed multivariate Student's t
        x = np.where(maskv, 0., x)
        mean = np.where(maskv, 0., mean)
        scale_masked = np.where(maskv + maskv.T, 0., scale)  # ensure masked entries are independent
        scale = np.where(np.diag(mask), 1, scale_masked)  # ensure masked entries return log like of 0

    log_det = 2 * np.sum(np.log(np.abs(np.diag(scale))))
    const = (
            gammaln((df + dim) * 0.5)
            - gammaln(df * 0.5)
            - 0.5 * dim * (np.log(df) + np.log(np.pi))
            - 0.5 * log_det
    )
    return const - 0.5 * (df + dim) * np.log(
        1.0 + (1.0 / df) * np.squeeze((x - mean).T @ solve(scale, (x - mean)))
    )


def pep_constant(var, power, mask=None):
    dim = var.shape[1]
    chol = cholesky(var, lower=True)
    log_diag_chol = np.log(np.abs(np.diag(chol)))

    if mask is not None:
        log_diag_chol = np.where(mask, 0., log_diag_chol)
        dim -= np.sum(np.array(mask, dtype=int))

    logdetvar = 2 * np.sum(log_diag_chol)
    constant = (
        0.5 * dim * ((1 - power) * LOG2PI - np.log(power))
        + 0.5 * (1 - power) * logdetvar
    )
    return constant


def mvn_logpdf_and_derivs(x, mean, cov, mask=None):
    """
    evaluate a multivariate Gaussian (log) pdf and compute its derivatives w.r.t. the mean
    """
    if mask is not None:
        # build a mask for computing the log likelihood of a partially observed multivariate Gaussian
        maskv = mask.reshape(-1, 1)
        mean = np.where(maskv, x, mean)
        cov_masked = np.where(maskv + maskv.T, 0., cov)  # ensure masked entries are independent
        cov = np.where(np.diag(mask), INV2PI, cov_masked)  # ensure masked entries return log like of 0

    n = mean.shape[0]
    cho, low = cho_factor(cov, lower=True)
    precision = cho_solve((cho, low), np.eye(cho.shape[1]))  # second derivative
    log_det = 2 * np.sum(np.log(np.abs(np.diag(cho))))
    diff = x - mean
    scaled_diff = precision @ diff  # first derivative
    distance = diff.T @ scaled_diff
    return np.squeeze(-0.5 * (distance + n * LOG2PI + log_det)), scaled_diff, -precision


@vmap
def _gaussian_expected_log_lik(y, post_mean, post_cov, var):
    post_mean = post_mean.reshape(-1, 1)
    post_cov = post_cov.reshape(-1, 1)
    y = y.reshape(-1, 1)
    var = var.reshape(-1, 1)
    # version which computes sum and outputs scalar
    # exp_log_lik = (
    #     -0.5 * y.shape[-2] * np.log(2 * np.pi)  # multiplier based on dimensions needed if taking sum of other terms
    #     - 0.5 * np.sum(np.log(var))
    #     - 0.5 * np.sum(((y - post_mean) ** 2 + post_cov) / var)
    # )
    # version which computes individual parts and outputs vector
    exp_log_lik = (
        -0.5 * np.log(2 * np.pi)
        - 0.5 * np.log(var)
        - 0.5 * ((y - post_mean) ** 2 + post_cov) / var
    )
    return exp_log_lik


def gaussian_expected_log_lik_diag(y, post_mean, post_cov, var):
    """
    Computes the "variational expectation", i.e. the
    expected log-likelihood, and its derivatives w.r.t. the posterior mean
        E[log 𝓝(yₙ|fₙ,σ²)] = ∫ log 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ
    :param y: data / observation (yₙ)
    :param post_mean: posterior mean (mₙ)
    :param post_cov: posterior variance (vₙ)
    :param var: variance, σ², of the Gaussian observation model p(yₙ|fₙ)=𝓝(yₙ|fₙ,σ²)
    :return:
        exp_log_lik: the expected log likelihood, E[log 𝓝(yₙ|fₙ,var)]  [scalar]
    """
    post_cov = diag(post_cov)
    var = diag(var)
    # var_exp = vmap(_gaussian_expected_log_lik)(y, post_mean, post_cov, var)
    var_exp = _gaussian_expected_log_lik(y, post_mean, post_cov, var)
    # return np.sum(var_exp)
    return var_exp


def gaussian_expected_log_lik(Y, q_mu, q_covar, noise, mask=None):
    """
    :param Y: N x 1
    :param q_mu: N x 1
    :param q_covar: N x N
    :param noise: N x N
    :param mask: N x 1
    :return:
        E[log 𝓝(yₙ|fₙ,σ²)] = ∫ log 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ
    """
    if mask is not None:
        # build a mask for computing the log likelihood of a partially observed multivariate Gaussian
        maskv = mask.reshape(-1, 1)
        q_mu = np.where(maskv, Y, q_mu)
        noise = np.where(maskv + maskv.T, 0., noise)  # ensure masked entries are independent
        noise = np.where(np.diag(mask), INV2PI, noise)  # ensure masked entries return log like of 0
        q_covar = np.where(maskv + maskv.T, 0., q_covar)  # ensure masked entries are independent
        q_covar = np.where(np.diag(mask), 1e-20, q_covar)  # ensure masked entries return trace term of 0

    ml = mvn_logpdf(Y, q_mu, noise)
    trace_term = -0.5 * np.trace(solve(noise, q_covar))
    return ml + trace_term


def compute_cavity(post_mean, post_cov, site_nat1, site_nat2, power, jitter=1e-8):
    """
    remove local likelihood approximation  from the posterior to obtain the marginal cavity distribution
    """
    post_nat2 = inv(post_cov + jitter * np.eye(post_cov.shape[1]))
    cav_cov = inv(post_nat2 - power * site_nat2)  # cavity covariance
    cav_mean = cav_cov @ (post_nat2 @ post_mean - power * site_nat1)  # cavity mean
    return cav_mean, cav_cov


def build_joint(ind, mean, cov, smoother_gain):
    """
    joint posterior (i.e. smoothed) mean and covariance of the states [u_, u+] at time t
    """
    mean_joint = np.block([[mean[ind]],
                           [mean[ind + 1]]])
    cross_cov = smoother_gain[ind] @ cov[ind + 1]
    cov_joint = np.block([[cov[ind], cross_cov],
                          [cross_cov.T, cov[ind + 1]]])
    return mean_joint, cov_joint


def set_z_stats(t, z):
    ind = (np.searchsorted(z.reshape(-1, ), t[:, :1].reshape(-1, )) - 1)
    num_neighbours = np.array([np.sum(ind == m) for m in range(z.shape[0] - 1)])
    return ind, num_neighbours


def gaussian_first_derivative_wrt_mean(f, m, C, w):
    invC = inv(C)
    return invC @ (f - m) * w


def gaussian_second_derivative_wrt_mean(f, m, C, w):
    invC = inv(C)
    return (invC @ (f - m) @ (f - m).T @ invC - invC) * w


def scaled_squared_euclid_dist(X, X2, ell):
    """
    Returns ‖(X - X2ᵀ) / ℓ‖², i.e. the squared L₂-norm.
    Adapted from GPflow: https://github.com/GPflow/GPflow
    """
    return square_distance(X / ell, X2 / ell)


def square_distance(X, X2):
    """
    Adapted from GPflow: https://github.com/GPflow/GPflow

    Returns ||X - X2ᵀ||²
    Due to the implementation and floating-point imprecision, the
    result may actually be very slightly negative for entries very
    close to each other.

    This function can deal with leading dimensions in X and X2.
    In the sample case, where X and X2 are both 2 dimensional,
    for example, X is [N, D] and X2 is [M, D], then a tensor of shape
    [N, M] is returned. If X is [N1, S1, D] and X2 is [N2, S2, D]
    then the output will be [N1, S1, N2, S2].
    """
    Xs = np.sum(np.square(X), axis=-1)
    X2s = np.sum(np.square(X2), axis=-1)
    dist = -2 * np.tensordot(X, X2, [[-1], [-1]])
    dist += broadcasting_elementwise(np.add, Xs, X2s)
    return dist


def broadcasting_elementwise(op, a, b):
    """
    Adapted from GPflow: https://github.com/GPflow/GPflow

    Apply binary operation `op` to every pair in tensors `a` and `b`.

    :param op: binary operator on tensors, e.g. tf.add, tf.substract
    :param a: tf.Tensor, shape [n_1, ..., n_a]
    :param b: tf.Tensor, shape [m_1, ..., m_b]
    :return: tf.Tensor, shape [n_1, ..., n_a, m_1, ..., m_b]
    """
    flatres = op(np.reshape(a, [-1, 1]), np.reshape(b, [1, -1]))
    return flatres.reshape(a.shape[0], b.shape[0])


def rotation_matrix(dt, omega):
    """
    Discrete time rotation matrix
    :param dt: step size [1]
    :param omega: frequency [1]
    :return:
        R: rotation matrix [2, 2]
    """
    R = np.array([
        [np.cos(omega * dt), -np.sin(omega * dt)],
        [np.sin(omega * dt),  np.cos(omega * dt)]
    ])
    return R


def balance(F: np.ndarray,
            iters: int) -> np.ndarray:
    dim = F.shape[0]
    d = np.ones((dim,))

    def loop_over_iters(carry_, _):
        F, d = carry_

        def loop_over_dims(carry, _):
            F, d, i = carry

            tmp = F[:, i]
            tmp = tmp.at[i].set(0.)
            c = np.linalg.norm(tmp, 2)
            tmp2 = F[i, :]
            tmp2 = tmp2.at[i].set(0.)

            r = np.linalg.norm(tmp2, 2)
            f = np.sqrt(r / c)
            d = d.at[i].set(d[i] * f)
            F = F.at[:,i].set(F[:, i] * f)
            F = F.at[i,:].set(F[i, :] / f)
            return (F, d, i+1), d

        (F, d, _), d_all = scan(f=loop_over_dims,
                                init=(F, d, 0),
                                xs=np.zeros(dim))

        return (F, d), d

    (_, d), _ = scan(f=loop_over_iters,
                     init=(F, d),
                     xs=np.zeros(iters))

    return d


# def bitmappify(ax, dpi=None):
#     """
#     # Convert vector axes content to raster (bitmap) images
#     :param ax: the figure axis
#     :param dpi: dots per inch
#     """
#     fig = ax.figure
#     # safe plot without axes
#     ax.set_axis_off()
#     fig.savefig('temp.png', dpi=dpi, transparent=True)
#     ax.set_axis_on()
#     # remember geometry
#     xl = ax.get_xlim()
#     yl = ax.get_ylim()
#     xb = ax.bbox._bbox.corners()[:, 0]
#     xb = (min(xb), max(xb))
#     yb = ax.bbox._bbox.corners()[:, 1]
#     yb = (min(yb), max(yb))
#     # compute coordinates to place bitmap image later
#     xb = (- xb[0] / (xb[1] - xb[0]), (1 - xb[0]) / (xb[1] - xb[0]))
#     xb = (xb[0] * (xl[1] - xl[0]) + xl[0], xb[1] * (xl[1] - xl[0]) + xl[0])
#     yb = (- yb[0] / (yb[1] - yb[0]), (1 - yb[0]) / (yb[1] - yb[0]))
#     yb = (yb[0] * (yl[1] - yl[0]) + yl[0], yb[1] * (yl[1] - yl[0]) + yl[0])
#     # replace the dots by the bitmap
#     del ax.collections[:]
#     del ax.lines[:]
#     ax.imshow(read_png('temp.png'), origin='upper',
#               aspect='auto', extent=(xb[0], xb[1], yb[0], yb[1]), label='_nolegend_')
#     # reset view
#     ax.set_xlim(xl)
#     ax.set_ylim(yl)
