import bayesnewton
import objax
import numpy as np
from jax.config import config
config.update("jax_enable_x64", True)
import pytest


def wiggly_time_series(x_):
    noise_var = 0.15  # true observation noise
    return (np.cos(0.04*x_+0.33*np.pi) * np.sin(0.2*x_) +
            np.sqrt(noise_var) * np.random.normal(0, 1, x_.shape))


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
    kernel = bayesnewton.kernels.Matern52(variance=var_f, lengthscale=len_f)
    likelihood = bayesnewton.likelihoods.Gaussian(variance=var_y)
    model = bayesnewton.models.VariationalGP(kernel=kernel, likelihood=likelihood, X=x, Y=y)
    return model


def initialise_markovgp_model(var_f, len_f, var_y, x, y):
    kernel = bayesnewton.kernels.Matern52(variance=var_f, lengthscale=len_f)
    likelihood = bayesnewton.likelihoods.Gaussian(variance=var_y)
    model = bayesnewton.models.MarkovVariationalGP(kernel=kernel, likelihood=likelihood, X=x, Y=y)
    return model


@pytest.mark.parametrize('var_f', [0.5, 1.5])
@pytest.mark.parametrize('len_f', [0.75, 2.5])
@pytest.mark.parametrize('var_y', [0.1, 0.5])
@pytest.mark.parametrize('N', [30, 60])
def test_initial_loss(var_f, len_f, var_y, N):
    """
    test whether VI with newt's GP and MarkovGP give the same initial ELBO and posterior
    """

    x, y = build_data(N)

    gp_model = initialise_gp_model(var_f, len_f, var_y, x, y)
    markovgp_model = initialise_markovgp_model(var_f, len_f, var_y, x, y)

    gp_model.update_posterior()
    loss_gp = gp_model.energy()
    print(loss_gp)

    markovgp_model.update_posterior()
    loss_markovgp = markovgp_model.energy()
    print(loss_markovgp)

    # print(posterior_mean - f_mean[:, 0])

    np.testing.assert_allclose(gp_model.posterior_mean.value, markovgp_model.posterior_mean.value, rtol=1e-4)
    np.testing.assert_allclose(gp_model.posterior_variance.value, markovgp_model.posterior_variance.value, rtol=1e-4)
    np.testing.assert_almost_equal(loss_gp, loss_markovgp, decimal=2)


@pytest.mark.parametrize('var_f', [0.5, 1.5])
@pytest.mark.parametrize('len_f', [0.75, 2.5])
@pytest.mark.parametrize('var_y', [0.1, 0.5])
@pytest.mark.parametrize('N', [30, 60])
def test_gradient_step(var_f, len_f, var_y, N):
    """
    test whether VI with newt's GP and MarkovGP provide the same initial gradient step in the hyperparameters
    """

    x, y = build_data(N)

    gp_model = initialise_gp_model(var_f, len_f, var_y, x, y)
    markovgp_model = initialise_markovgp_model(var_f, len_f, var_y, x, y)

    gv = objax.GradValues(gp_model.energy, gp_model.vars())
    gv_markov = objax.GradValues(markovgp_model.energy, markovgp_model.vars())

    lr_adam = 0.1
    lr_newton = 1.
    opt = objax.optimizer.Adam(gp_model.vars())
    opt_markov = objax.optimizer.Adam(markovgp_model.vars())

    gp_model.update_posterior()
    gp_grads, gp_value = gv()
    gp_loss_ = gp_value[0]
    opt(lr_adam, gp_grads)
    gp_hypers = np.array([gp_model.kernel.lengthscale, gp_model.kernel.variance, gp_model.likelihood.variance])
    print(gp_hypers)
    print(gp_grads)

    markovgp_model.update_posterior()
    markovgp_grads, markovgp_value = gv_markov()
    markovgp_loss_ = markovgp_value[0]
    opt_markov(lr_adam, markovgp_grads)
    markovgp_hypers = np.array([markovgp_model.kernel.lengthscale, markovgp_model.kernel.variance,
                                markovgp_model.likelihood.variance])
    print(markovgp_hypers)
    print(markovgp_grads)

    np.testing.assert_allclose(gp_grads[0], markovgp_grads[0], rtol=1e-4)
    np.testing.assert_allclose(gp_grads[1], markovgp_grads[1], rtol=1e-4)
    np.testing.assert_allclose(gp_grads[2], markovgp_grads[2], rtol=1e-4)


@pytest.mark.parametrize('var_f', [0.5, 1.5])
@pytest.mark.parametrize('len_f', [0.75, 2.5])
@pytest.mark.parametrize('var_y', [0.1, 0.5])
@pytest.mark.parametrize('N', [30, 60])
def test_inference_step(var_f, len_f, var_y, N):
    """
    test whether VI with newt's GP and MarkovGP give the same posterior after one natural gradient step
    """

    x, y = build_data(N)

    gp_model = initialise_gp_model(var_f, len_f, var_y, x, y)
    markovgp_model = initialise_markovgp_model(var_f, len_f, var_y, x, y)

    lr_newton = 1.

    gp_model.inference(lr=lr_newton)  # update variational params

    markovgp_model.inference(lr=lr_newton)  # update variational params

    np.testing.assert_allclose(gp_model.posterior_mean.value, markovgp_model.posterior_mean.value, rtol=1e-4)
    np.testing.assert_allclose(gp_model.posterior_variance.value, markovgp_model.posterior_variance.value, rtol=1e-4)
