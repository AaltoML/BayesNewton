import bayesnewton
import numpy as np
from bayesnewton.utils import solve
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


@pytest.mark.parametrize('var_f', [0.5, 1.5])
@pytest.mark.parametrize('len_f', [0.75, 2.5])
@pytest.mark.parametrize('var_y', [0.1, 0.5])
@pytest.mark.parametrize('N', [30, 60])
def test_marg_lik(var_f, len_f, var_y, N):
    """
    test whether VI with newt's GP and Gaussian likelihood gives the exact marginal likelihood
    """

    x, y = build_data(N)

    gp_model = initialise_gp_model(var_f, len_f, var_y, x, y)

    gp_model.inference(lr=1.)  # update variational params
    loss_gp = gp_model.energy()
    print(loss_gp)

    K_X = gp_model.kernel(x, x)
    K_Y = K_X + var_y * np.eye(K_X.shape[0])
    L_Y = np.linalg.cholesky(K_Y)
    exact_marg_lik = (
            -0.5 * y.T @ solve(K_Y, y)
            - np.sum(np.log(np.diag(L_Y)))
            - 0.5 * y.shape[0] * np.log(2 * np.pi)
    )

    print(exact_marg_lik)

    np.testing.assert_almost_equal(loss_gp, -exact_marg_lik, decimal=4)
