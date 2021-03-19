import newt
import objax
import numpy as np
from jax import vmap
from newt.utils import compute_measurement
import matplotlib.pyplot as plt
from jax.config import config
config.update("jax_enable_x64", True)
import pytest


inf = newt.inference.VariationalInference()


def wiggly_time_series(x_):
    noise_var = 0.15  # true observation noise
    return (np.cos(0.04*x_+0.33*np.pi) * np.sin(0.2*x_) +
            np.math.sqrt(noise_var) * np.random.normal(0, 1, x_.shape))


def build_data(N):
    # np.random.seed(12345)
    x = np.random.permutation(np.linspace(-25.0, 150.0, num=N) + 0.5*np.random.randn(N))  # unevenly spaced
    x = np.sort(x)  # since MarkovGP sorts the inputs, they must also be sorted for GP
    y = wiggly_time_series(x)
    # x_test = np.linspace(np.min(x)-15.0, np.max(x)+15.0, num=500)
    # y_test = wiggly_time_series(x_test)
    # x_plot = np.linspace(np.min(x)-20.0, np.max(x)+20.0, 200)

    x = x[:, None]
    # y = y[:, None]
    # x_plot = x_plot[:, None]
    return x, y


def initialise_gp_model(var_f, len_f, var_y, x, y):
    kernel = newt.kernels.Matern52(variance=var_f, lengthscale=len_f)
    likelihood = newt.likelihoods.Gaussian(variance=var_y)
    model = newt.models.GP(kernel=kernel, likelihood=likelihood, X=x, Y=y)
    return model


def initialise_sparsemarkovgp_model(var_f, len_f, var_y, x, y, z=None):
    if z is None:
        z = x
    kernel = newt.kernels.Matern52(variance=var_f, lengthscale=len_f)
    likelihood = newt.likelihoods.Gaussian(variance=var_y)
    model = newt.models.SparseMarkovGP(kernel=kernel, likelihood=likelihood, X=x, Y=y, Z=z)
    return model


@pytest.mark.parametrize('var_f', [0.5, 1.5])
@pytest.mark.parametrize('len_f', [4.5, 7.5])
@pytest.mark.parametrize('var_y', [0.1, 0.5])
@pytest.mark.parametrize('N', [30, 60])
def test_initial_loss(var_f, len_f, var_y, N):
    """
    test whether MarkovGP and SparseMarkovGP give the same initial ELBO and posterior (Z=X)
    """

    x, y = build_data(N)

    gp_model = initialise_gp_model(var_f, len_f, var_y, x, y)
    sparsemarkovgp_model = initialise_sparsemarkovgp_model(var_f, len_f, var_y, x, y)

    gp_model.update_posterior()
    loss_gp = inf(gp_model)
    print(loss_gp)

    sparsemarkovgp_model.update_posterior()
    loss_markovgp = inf(sparsemarkovgp_model)
    print(loss_markovgp)

    # print(posterior_mean - f_mean[:, 0])

    # measure_func = vmap(
    #     compute_measurement, (None, 0, 0, 0)
    # )
    # post_mean, post_cov = measure_func(sparsemarkovgp_model.kernel,
    #                                    sparsemarkovgp_model.Z[:-1, :1],
    #                                    sparsemarkovgp_model.posterior_mean.value,
    #                                    sparsemarkovgp_model.posterior_variance.value)
    post_mean = sparsemarkovgp_model.posterior_mean.value[1:, :1, :1]
    post_cov = sparsemarkovgp_model.posterior_variance.value[1:, :1, :1]

    np.testing.assert_allclose(gp_model.posterior_mean.value, post_mean, rtol=1e-4)
    np.testing.assert_allclose(gp_model.posterior_variance.value, post_cov, rtol=1e-4)
    np.testing.assert_almost_equal(loss_gp, loss_markovgp, decimal=2)


@pytest.mark.parametrize('var_f', [0.5, 1.5])
@pytest.mark.parametrize('len_f', [4.5, 7.5])
@pytest.mark.parametrize('var_y', [0.1, 0.5])
@pytest.mark.parametrize('N', [30, 60])
def test_gradient_step(var_f, len_f, var_y, N):
    """
    test whether MarkovGP and SparseMarkovGP provide the same initial gradient step in the hyperparameters (Z=X)
    """

    x, y = build_data(N)

    gp_model = initialise_gp_model(var_f, len_f, var_y, x, y)
    sparsemarkovgp_model = initialise_sparsemarkovgp_model(var_f, len_f, var_y, x, y)

    gv = objax.GradValues(inf, gp_model.vars())
    gv_markov = objax.GradValues(inf, sparsemarkovgp_model.vars())

    lr_adam = 0.1
    lr_newton = 1.
    opt = objax.optimizer.Adam(gp_model.vars())
    opt_markov = objax.optimizer.Adam(sparsemarkovgp_model.vars())

    gp_model.update_posterior()
    gp_grads, gp_value = gv(gp_model, lr=lr_newton)
    gp_loss_ = gp_value[0]
    opt(lr_adam, gp_grads)
    gp_hypers = np.array([gp_model.kernel.lengthscale, gp_model.kernel.variance, gp_model.likelihood.variance])
    print(gp_hypers)
    print(gp_grads)

    sparsemarkovgp_model.update_posterior()
    markovgp_grads, markovgp_value = gv_markov(sparsemarkovgp_model, lr=lr_newton)
    markovgp_loss_ = markovgp_value[0]
    opt_markov(lr_adam, markovgp_grads)
    markovgp_hypers = np.array([sparsemarkovgp_model.kernel.lengthscale, sparsemarkovgp_model.kernel.variance,
                                sparsemarkovgp_model.likelihood.variance])
    print(markovgp_hypers)
    print(markovgp_grads)

    np.testing.assert_allclose(gp_grads[0], markovgp_grads[0], atol=1e-3)
    np.testing.assert_allclose(gp_grads[1], markovgp_grads[1], rtol=1e-4)
    np.testing.assert_allclose(gp_grads[2], markovgp_grads[2], rtol=1e-4)


@pytest.mark.parametrize('var_f', [0.5, 1.5])
@pytest.mark.parametrize('len_f', [4.5, 7.5])
@pytest.mark.parametrize('var_y', [0.1, 0.5])
@pytest.mark.parametrize('N', [30, 60])
def test_inference_step(var_f, len_f, var_y, N):
    """
    test whether MarkovGP and SparseMarkovGP give the same posterior after one natural gradient step (Z=X)
    """

    x, y = build_data(N)

    gp_model = initialise_gp_model(var_f, len_f, var_y, x, y)
    sparsemarkovgp_model = initialise_sparsemarkovgp_model(var_f, len_f, var_y, x, y)

    lr_newton = 1.

    gp_model.update_posterior()
    gp_loss = inf(gp_model, lr=lr_newton)  # update variational params
    gp_model.update_posterior()

    sparsemarkovgp_model.update_posterior()
    markovgp_loss = inf(sparsemarkovgp_model, lr=lr_newton)  # update variational params
    sparsemarkovgp_model.update_posterior()

    # measure_func = vmap(
    #     compute_measurement, (None, 0, 0, 0)
    # )
    # post_mean, post_cov = measure_func(sparsemarkovgp_model.kernel,
    #                                    sparsemarkovgp_model.Z[:-1, :1],
    #                                    sparsemarkovgp_model.posterior_mean.value,
    #                                    sparsemarkovgp_model.posterior_variance.value)
    post_mean = sparsemarkovgp_model.posterior_mean.value[1:, :1, :1]
    post_cov = sparsemarkovgp_model.posterior_variance.value[1:, :1, :1]

    np.testing.assert_allclose(gp_model.posterior_mean.value, post_mean, rtol=1e-4)
    np.testing.assert_allclose(gp_model.posterior_variance.value, post_cov, rtol=1e-4)


@pytest.mark.parametrize('var_f', [0.5, 1.5])
@pytest.mark.parametrize('len_f', [4.5, 7.5])
@pytest.mark.parametrize('var_y', [0.1, 0.5])
@pytest.mark.parametrize('N', [30, 60])
def test_initial_loss(var_f, len_f, var_y, N):
    """
    test whether MarkovGP and SparseMarkovGP give the same initial ELBO and posterior (Z=X)
    """

    x, y = build_data(N)

    z = x + np.random.normal(0, .05, x.shape)

    gp_model = initialise_gp_model(var_f, len_f, var_y, x, y)
    sparsemarkovgp_model = initialise_sparsemarkovgp_model(var_f, len_f, var_y, x, y, z)

    gp_model.update_posterior()
    loss_gp = inf(gp_model)
    print(loss_gp)

    sparsemarkovgp_model.update_posterior()
    loss_markovgp = inf(sparsemarkovgp_model)
    print(loss_markovgp)

    # print(posterior_mean - f_mean[:, 0])

    post_mean = sparsemarkovgp_model.posterior_mean.value[1:, :1, :1]
    post_cov = sparsemarkovgp_model.posterior_variance.value[1:, :1, :1]

    np.testing.assert_allclose(gp_model.posterior_mean.value, post_mean, rtol=1e-1)
    # np.testing.assert_allclose(gp_model.posterior_variance.value, post_cov, rtol=1e-4)
    np.testing.assert_almost_equal(loss_gp, loss_markovgp, decimal=-1)


@pytest.mark.parametrize('var_f', [0.5, 1.5])
@pytest.mark.parametrize('len_f', [4.5, 7.5])
@pytest.mark.parametrize('var_y', [0.1, 0.5])
@pytest.mark.parametrize('N', [30, 60])
def test_gradient_step(var_f, len_f, var_y, N):
    """
    test whether MarkovGP and SparseMarkovGP provide the same initial gradient step in the hyperparameters (Z=X)
    """

    x, y = build_data(N)

    z = x + np.random.normal(0, .05, x.shape)

    gp_model = initialise_gp_model(var_f, len_f, var_y, x, y)
    sparsemarkovgp_model = initialise_sparsemarkovgp_model(var_f, len_f, var_y, x, y, z)

    gv = objax.GradValues(inf, gp_model.vars())
    gv_markov = objax.GradValues(inf, sparsemarkovgp_model.vars())

    lr_adam = 0.1
    lr_newton = 1.
    opt = objax.optimizer.Adam(gp_model.vars())
    opt_markov = objax.optimizer.Adam(sparsemarkovgp_model.vars())

    gp_model.update_posterior()
    gp_grads, gp_value = gv(gp_model, lr=lr_newton)
    gp_loss_ = gp_value[0]
    opt(lr_adam, gp_grads)
    gp_hypers = np.array([gp_model.kernel.lengthscale, gp_model.kernel.variance, gp_model.likelihood.variance])
    print(gp_hypers)
    print(gp_grads)

    sparsemarkovgp_model.update_posterior()
    markovgp_grads, markovgp_value = gv_markov(sparsemarkovgp_model, lr=lr_newton)
    markovgp_loss_ = markovgp_value[0]
    opt_markov(lr_adam, markovgp_grads)
    markovgp_hypers = np.array([sparsemarkovgp_model.kernel.lengthscale, sparsemarkovgp_model.kernel.variance,
                                sparsemarkovgp_model.likelihood.variance])
    print(markovgp_hypers)
    print(markovgp_grads)

    np.testing.assert_allclose(gp_grads[0], markovgp_grads[0], atol=3e-1)
    np.testing.assert_allclose(gp_grads[1], markovgp_grads[1], rtol=5e-2)
    np.testing.assert_allclose(gp_grads[2], markovgp_grads[2], rtol=5e-2)


@pytest.mark.parametrize('var_f', [0.5, 1.5])
@pytest.mark.parametrize('len_f', [4.5, 7.5])
@pytest.mark.parametrize('var_y', [0.1, 0.5])
@pytest.mark.parametrize('N', [30, 60])
def test_inference_step(var_f, len_f, var_y, N):
    """
    test whether MarkovGP and SparseMarkovGP give almost the same posterior after one
    step when Z is a perturbed version of X
    """

    x, y = build_data(N)

    z = x + np.random.normal(0, 0.5, x.shape)

    gp_model = initialise_gp_model(var_f, len_f, var_y, x, y)
    sparsemarkovgp_model = initialise_sparsemarkovgp_model(var_f, len_f, var_y, x, y, z)

    lr_newton = 1.

    gp_model.update_posterior()
    gp_loss = inf(gp_model, lr=lr_newton)  # update variational params
    gp_model.update_posterior()

    sparsemarkovgp_model.update_posterior()
    markovgp_loss = inf(sparsemarkovgp_model, lr=lr_newton)  # update variational params
    sparsemarkovgp_model.update_posterior()

    post_mean, post_cov = sparsemarkovgp_model(np.sort(x))

    # post_mean = sparsemarkovgp_model.posterior_mean.value[1:, :1, :1]
    # post_cov = sparsemarkovgp_model.posterior_variance.value[1:, :1, :1]

    np.testing.assert_allclose(np.squeeze(gp_model.posterior_mean.value), post_mean, atol=1e-1)
    np.testing.assert_allclose(np.squeeze(gp_model.posterior_variance.value), post_cov, atol=1e-1)


# N = 30
# var_f = .5
# len_f = 5.
# var_y = 0.1
#
# np.random.seed(123)
# x, y = build_data(N)
#
# z = x + np.random.normal(0., 1, x.shape)
# # z[-1] += 5
# # z[-2] -= 1
# # z[9] -= 1
# # z = np.concatenate([z, np.array([z[9]+1])], axis=0)
# # z -= np.abs(np.random.normal(0, .1, x.shape))
#
# gp_model = initialise_gp_model(var_f, len_f, var_y, x, y)
# sparsemarkovgp_model = initialise_sparsemarkovgp_model(var_f, len_f, var_y, x, y, z)
#
# lr_newton = 1.
#
# gp_model.update_posterior()
# gp_loss = inf(gp_model, lr=lr_newton)  # update variational params
# gp_model.update_posterior()
#
# sparsemarkovgp_model.update_posterior()
# markovgp_loss = inf(sparsemarkovgp_model, lr=lr_newton)  # update variational params
# sparsemarkovgp_model.update_posterior()
#
# post_mean, post_cov = sparsemarkovgp_model(np.sort(x))
# post_mean_ = sparsemarkovgp_model.posterior_mean.value[:, 0, 0]
#
# plt.plot(x, y, 'k.')
# plt.plot(z, np.zeros_like(z), 'b.')
# plt.plot(x, post_mean, 'b-')
# plt.plot(x, post_mean + np.sqrt(post_cov), 'b-', alpha=0.4)
# plt.plot(x, post_mean - np.sqrt(post_cov), 'b-', alpha=0.4)
# plt.plot(x, np.squeeze(gp_model.posterior_mean.value), 'r--')
# plt.plot(x, np.squeeze(gp_model.posterior_mean.value) + np.sqrt(np.squeeze(gp_model.posterior_variance.value)), 'r--', alpha=0.4)
# plt.plot(x, np.squeeze(gp_model.posterior_mean.value) - np.sqrt(np.squeeze(gp_model.posterior_variance.value)), 'r--', alpha=0.4)
# plt.show()
#
# # post_mean = sparsemarkovgp_model.posterior_mean.value[1:, :1, :1]
# # post_cov = sparsemarkovgp_model.posterior_variance.value[1:, :1, :1]
#
# np.testing.assert_allclose(np.squeeze(gp_model.posterior_mean.value), post_mean, atol=1e-1)
# np.testing.assert_allclose(np.squeeze(gp_model.posterior_variance.value), post_cov, atol=1e-1)
