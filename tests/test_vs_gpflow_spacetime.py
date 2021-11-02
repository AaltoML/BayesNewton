import bayesnewton
import objax
import numpy as np
from bayesnewton.utils import inv
import gpflow
from jax.config import config
config.update("jax_enable_x64", True)
import pytest
import scipy as sp

gpflow.config.set_default_jitter(1e-32)

# TODO: ------- FIX --------


# def inv_(K):
#     K_chol = sp.linalg.cholesky(K, lower=True)
#     return sp.linalg.cho_solve((K_chol, True), np.eye(K.shape[0]))


def create_grid(x1, x2, y1, y2, n1=10, n2=10):
    y_ = np.linspace(y1, y2, n2)
    x_ = np.linspace(x1, x2, n1)

    grid = []
    for i in x_:
        for j in y_:
            grid.append([i, j])

    return np.array(grid)


def build_data(N_):
    Nt_train = N_
    Ns = N_
    X_ = create_grid(0, 1, 0, 1, Nt_train, Ns)
    t_ = np.linspace(0, 1, Nt_train, dtype=float)
    R_ = np.tile(np.linspace(0, 1, Ns, dtype=float)[None, ...], [Nt_train, 1])

    y_ = np.sin(10 * X_[:, 0]) + np.sin(10 * X_[:, 1]) + 0.01 * np.random.randn(X_.shape[0])

    # Y = y[:, None]
    Y_ = y_.reshape(Nt_train, Ns)
    return X_, Y_, t_, R_, y_


def initialise_gp_model(var_f, len_f, var_y, x_, y_, z_):
    kernel = bayesnewton.kernels.SpatialMatern52(variance=var_f, lengthscale=len_f,
                                          z=z_, sparse=True, opt_z=False, conditional='Full')
    likelihood = bayesnewton.likelihoods.Gaussian(variance=var_y)

    # the sort during utils.input_admin() sometimes results in different sorting of inputs
    # so this step ensures everything is aligned
    # model_ = bayesnewton.models.MarkovGP(kernel=kernel, likelihood=likelihood, X=x_, Y=y_)
    # x_sorted = model_.X
    # r_sorted = model_.R
    # x_ = np.vstack([x_sorted.T, r_sorted.T]).T
    # y_ = model_.Y

    model = bayesnewton.models.VariationalGP(kernel=kernel, likelihood=likelihood, X=x_, Y=y_)
    return model


def initialise_markovgp_model(var_f, len_f, var_y, x_, y_, z_):
    kernel = bayesnewton.kernels.SpatialMatern52(variance=var_f, lengthscale=len_f,
                                                 z=z_, sparse=True, opt_z=False, conditional='Full')
    likelihood = bayesnewton.likelihoods.Gaussian(variance=var_y)
    model = bayesnewton.models.MarkovVariationalGP(kernel=kernel, likelihood=likelihood, X=x_, Y=y_)
    return model


def initialise_gpflow_model(var_f_, len_f_, var_y_, x_, y_):
    N_ = x_.shape[0]
    k0 = gpflow.kernels.Matern52(lengthscales=[len_f_], variance=var_f_, active_dims=[0], name='matern1')
    k1 = gpflow.kernels.Matern52(lengthscales=[len_f_], variance=1., active_dims=[1], name='matern2')
    k = k0 * k1

    # find the m and S that correspond to the same natural parameters used by CVI
    K_xx = np.array(k(x_, x_))
    K_xx_inv = inv(K_xx)

    # print(x_.shape)

    S = inv(K_xx_inv + 1e-2 * np.eye(N_))
    S_chol = np.linalg.cholesky(S)
    S_chol_init = np.array([S_chol])
    # print(np.diag(S))
    # S_chol_flattened_init = np.array(S_chol[np.tril_indices(N, 0)])

    lambda_init = np.zeros((N_, 1))
    m_init = S @ lambda_init

    lik = gpflow.likelihoods.Gaussian(variance=var_y_)

    # data = (x, y)
    # print(x)

    model = gpflow.models.SVGP(
        inducing_variable=x_,
        whiten=False,
        kernel=k,
        mean_function=None,
        likelihood=lik,
        q_mu=m_init,
        q_sqrt=S_chol_init
    )
    gpflow.utilities.set_trainable(model.inducing_variable.Z, False)
    gpflow.utilities.set_trainable(model.q_mu, False)
    gpflow.utilities.set_trainable(model.q_sqrt, False)
    return model


@pytest.mark.parametrize('var_f_', [0.5, 1.5])
@pytest.mark.parametrize('len_f_', [0.5, 1.])
@pytest.mark.parametrize('var_y_', [0.1, 0.5])
@pytest.mark.parametrize('N_', [8, 16])
def test_initial_loss(var_f_, len_f_, var_y_, N_):
    """
    test whether VI with newt's GP and MarkovGP give the same initial ELBO and posterior
    """

    x_, Y_, t_, R_, y_ = build_data(N_)

    gp_model = initialise_gp_model(var_f_, len_f_, var_y_, x_, y_, R_[0])
    markovgp_model = initialise_markovgp_model(var_f_, len_f_, var_y_, x_, y_, R_[0])
    gpflow_model = initialise_gpflow_model(var_f_, len_f_, var_y_, x_, y_)

    gp_model.update_posterior()
    f_mean_gp, f_var_gp = gp_model.predict(x_)
    loss_gp = gp_model.energy()
    print(loss_gp)

    markovgp_model.update_posterior()
    f_mean_markovgp, f_var_markovgp = markovgp_model.predict(x_)
    loss_markovgp = markovgp_model.energy()
    print(loss_markovgp)
    data = (x_, y_[..., None])
    f_mean_gpflow, f_var_gpflow = gpflow_model.predict_f(x_[None], full_cov=False, full_output_cov=False)
    # var_exp = np.sum(gpflow_model.likelihood.variational_expectations(f_mean, f_var, Y))
    loss_gpflow = -gpflow_model.elbo(data)
    print(loss_gpflow.numpy())

    # print(gp_model.posterior_variance.value - markovgp_model.posterior_variance.value)

    # np.testing.assert_allclose(gp_model.posterior_mean.value, markovgp_model.posterior_mean.value, rtol=1e-4)
    # np.testing.assert_allclose(gp_model.posterior_variance.value, markovgp_model.posterior_variance.value, rtol=1e-4)
    np.testing.assert_almost_equal(loss_gp, loss_markovgp, decimal=2)
    np.testing.assert_almost_equal(loss_gp, loss_gpflow.numpy(), decimal=2)
    np.testing.assert_allclose(f_var_gp, f_var_markovgp, rtol=1e-4)
    np.testing.assert_allclose(f_var_gp, np.squeeze(f_var_gpflow), rtol=1e-4)


# @pytest.mark.parametrize('var_f', [0.5, 1.5])
# @pytest.mark.parametrize('len_f', [0.75, 1.25])
# @pytest.mark.parametrize('var_y', [0.1])  # , 0.05])
# @pytest.mark.parametrize('N', [8])  # , 16])
# def test_gradient_step(var_f, len_f, var_y, N):
#     """
#     test whether VI with newt's GP and MarkovGP provide the same initial gradient step in the hyperparameters
#     """
#
#     x, Y, t, R, y = build_data(N)
#
#     gp_model = initialise_gp_model(var_f, len_f, var_y, x, y, R[0])
#     markovgp_model = initialise_markovgp_model(var_f, len_f, var_y, x, y, R[0])
#     gpflow_model = initialise_gpflow_model(var_f, len_f, var_y, x, y)
#
#     gv = objax.GradValues(inf, gp_model.vars())
#     gv_markov = objax.GradValues(inf, markovgp_model.vars())
#
#     lr_adam = 0.1
#     lr_newton = 1.
#     opt = objax.optimizer.Adam(gp_model.vars())
#     opt_markov = objax.optimizer.Adam(markovgp_model.vars())
#
#     gp_model.update_posterior()
#     gp_grads, gp_value = gv(gp_model, lr=lr_newton)
#     gp_loss_ = gp_value[0]
#     opt(lr_adam, gp_grads)
#     gp_hypers = np.array([gp_model.kernel.temporal_kernel.lengthscale,
#                           gp_model.kernel.temporal_kernel.variance,
#                           gp_model.kernel.spatial_kernel.lengthscale,
#                           gp_model.likelihood.variance])
#     print(gp_hypers)
#     print(gp_grads)
#
#     markovgp_model.update_posterior()
#     markovgp_grads, markovgp_value = gv_markov(markovgp_model, lr=lr_newton)
#     markovgp_loss_ = markovgp_value[0]
#     opt_markov(lr_adam, markovgp_grads)
#     markovgp_hypers = np.array([markovgp_model.kernel.temporal_kernel.lengthscale,
#                                 markovgp_model.kernel.temporal_kernel.variance,
#                                 markovgp_model.kernel.spatial_kernel.lengthscale,
#                                 markovgp_model.likelihood.variance])
#
#     print(markovgp_hypers)
#     print(markovgp_grads)
#
#     np.testing.assert_allclose(gp_grads[0], markovgp_grads[0], rtol=1e-4)
#     np.testing.assert_allclose(gp_grads[1], markovgp_grads[1], rtol=1e-4)
#     np.testing.assert_allclose(gp_grads[2], markovgp_grads[2], rtol=1e-4)


# @pytest.mark.parametrize('var_f', [0.5, 1.5])
# @pytest.mark.parametrize('len_f', [0.75, 2.5])
# @pytest.mark.parametrize('var_y', [0.1, 0.5])
# @pytest.mark.parametrize('N', [8, 16])
# def test_inference_step(var_f, len_f, var_y, N):
#     """
#     test whether VI with newt's GP and MarkovGP give the same posterior after one natural gradient step
#     """
#
#     x, Y, t, R, y = build_data(N)
#
#     gp_model = initialise_gp_model(var_f, len_f, var_y, x, y, R[0])
#     markovgp_model = initialise_markovgp_model(var_f, len_f, var_y, x, y, R[0])
#
#     lr_newton = 1.
#
#     gp_model.update_posterior()
#     gp_loss = inf(gp_model, lr=lr_newton)  # update variational params
#     gp_model.update_posterior()
#
#     markovgp_model.update_posterior()
#     markovgp_loss = inf(markovgp_model, lr=lr_newton)  # update variational params
#     markovgp_model.update_posterior()
#
#     np.testing.assert_allclose(gp_model.posterior_mean.value, markovgp_model.posterior_mean.value, rtol=1e-4)
#     np.testing.assert_allclose(gp_model.posterior_variance.value, markovgp_model.posterior_variance.value, rtol=1e-4)

# N = 5
# x, Y, t, R, y = build_data(N)
#
# var_f = 0.5
# len_f = 0.75
# var_y = 0.1
#
# gp_model = initialise_gp_model(var_f, len_f, var_y, x, y, R[0])
# markovgp_model = initialise_markovgp_model(var_f, len_f, var_y, x, y, R[0])
# gpflow_model = initialise_gpflow_model(var_f, len_f, var_y, x, y)
#
# lr_newton = 1.
#
# gp_model.update_posterior()
# f_mean_gp, f_var_gp = gp_model.predict(x)
# gp_loss = inf(gp_model, lr=lr_newton)  # update variational params
# print(gp_loss)
# # gp_model.update_posterior()
#
# markovgp_model.update_posterior()
# f_mean_markovgp, f_var_markobgp = markovgp_model.predict(x)
# markovgp_loss = inf(markovgp_model, lr=lr_newton)  # update variational params
# print(markovgp_loss)
# # markovgp_model.update_posterior()
#
# data = (x, y[:, None])
# # f_mean_gpflow, f_var_gpflow = gpflow_model.predict_f(x)
# f_mean_gpflow, f_var_gpflow = gpflow_model.predict_f(x[None], full_cov=False, full_output_cov=False)
# # var_exp = np.sum(gpflow_model.likelihood.variational_expectations(f_mean_gpflow, f_var_gpflow, y[:, None]))
# loss_gpflow = -gpflow_model.elbo(data)
# print(loss_gpflow.numpy())
#
# np.testing.assert_almost_equal(gp_loss, markovgp_loss, decimal=2)
# np.testing.assert_almost_equal(gp_loss, loss_gpflow.numpy(), decimal=2)
#
# np.testing.assert_allclose(gp_model.posterior_variance.value, markovgp_model.posterior_variance.value, rtol=1e-4)
#
# np.testing.assert_allclose(f_var_gp, np.squeeze(f_var_gpflow), rtol=1e-4)
