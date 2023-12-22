import bayesnewton
import objax
from bayesnewton.utils import inv
import numpy as np
from jax.config import config
config.update("jax_enable_x64", True)
import pytest
import tensorflow as tf
import gpflow

# TODO: ------- FIX --------


def build_data(N):
    # np.random.seed(12345)
    x = 100 * np.random.rand(N)
    f = lambda x_: 6 * np.sin(np.pi * x_ / 10.0) / (np.pi * x_ / 10.0 + 1)
    y_ = f(x) + np.sqrt(0.05) * np.random.randn(x.shape[0])
    y = np.sign(y_)
    y[y == -1] = 0
    x = x[:, None]
    return x, y


def initialise_newt_model(var_f, len_f, x, y):
    kernel = bayesnewton.kernels.Matern52(variance=var_f, lengthscale=len_f)
    likelihood = bayesnewton.likelihoods.Bernoulli()
    model = bayesnewton.models.VariationalGP(kernel=kernel, likelihood=likelihood, X=x, Y=y)
    return model


def initialise_gpflow_model(var_f, len_f, x, y):
    N = x.shape[0]
    k = gpflow.kernels.Matern52(lengthscales=[len_f], variance=var_f, name='matern')

    # find the m and S that correspond to the same natural parameters used by CVI
    K_xx = np.array(k(x, x))
    K_xx_inv = inv(K_xx)

    S = inv(K_xx_inv + 1e-2 * np.eye(N))
    S_chol = np.linalg.cholesky(S)
    S_chol_init = np.array([S_chol])
    # S_chol_flattened_init = np.array(S_chol[np.tril_indices(N, 0)])

    lambda_init = np.zeros((N, 1))
    m_init = S @ lambda_init

    lik = gpflow.likelihoods.Bernoulli()

    # data = (x, y)
    model = gpflow.models.SVGP(
        inducing_variable=x,
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


@pytest.mark.parametrize('var_f', [0.5, 1.5])
@pytest.mark.parametrize('len_f', [2.5, 5.])
@pytest.mark.parametrize('N', [30, 60])
def test_initial_loss(var_f, len_f, N):
    """
    test whether newt's VI and gpflow's SVGP (Z=X) give the same initial ELBO and posterior
    """

    x, y = build_data(N)

    newt_model = initialise_newt_model(var_f, len_f, x, y)
    gpflow_model = initialise_gpflow_model(var_f, len_f, x, y)

    newt_model.update_posterior()
    loss_newt = newt_model.energy()
    # _, _, expected_density = newt_model.inference(newt_model)
    print(loss_newt)
    # print(expected_density)

    data = (x, y[:, None])
    f_mean, f_var = gpflow_model.predict_f(x)
    var_exp = np.sum(gpflow_model.likelihood.variational_expectations(f_mean, f_var, y[:, None]))
    loss_gpflow = -gpflow_model.elbo(data)
    print(loss_gpflow.numpy())
    # print(var_exp)

    # print(posterior_mean - f_mean[:, 0])

    np.testing.assert_allclose(np.squeeze(newt_model.posterior_mean.value), f_mean[:, 0], rtol=1e-4)
    np.testing.assert_allclose(np.squeeze(newt_model.posterior_variance.value), f_var[:, 0], rtol=1e-4)
    np.testing.assert_almost_equal(loss_newt, loss_gpflow.numpy(), decimal=2)


@pytest.mark.parametrize('var_f', [0.5, 1.5])
@pytest.mark.parametrize('len_f', [2.5, 5.])
@pytest.mark.parametrize('N', [30, 60])
def test_gradient_step(var_f, len_f, N):
    """
    test whether newt's VI and gpflow's SVGP (Z=X) provide the same initial gradient step in the hyperparameters
    """

    x, y = build_data(N)

    newt_model = initialise_newt_model(var_f, len_f, x, y)
    gpflow_model = initialise_gpflow_model(var_f, len_f, x, y)

    gv = objax.GradValues(newt_model.energy, newt_model.vars())

    lr_adam = 0.1
    lr_newton = 1.
    opt = objax.optimizer.Adam(newt_model.vars())

    newt_model.update_posterior()
    newt_grads, value = gv()  # , lr=lr_newton)
    loss_ = value[0]
    opt(lr_adam, newt_grads)
    newt_hypers = np.array([newt_model.kernel.lengthscale, newt_model.kernel.variance])
    print(newt_hypers)
    print(newt_grads)

    adam_opt = tf.optimizers.Adam(lr_adam)
    data = (x, y[:, None])
    with tf.GradientTape() as tape:
        loss = -gpflow_model.elbo(data)
    _vars = gpflow_model.trainable_variables
    gpflow_grads = tape.gradient(loss, _vars)

    loss_fn = gpflow_model.training_loss_closure(data)
    adam_vars = gpflow_model.trainable_variables
    adam_opt.minimize(loss_fn, adam_vars)
    gpflow_hypers = np.array([gpflow_model.kernel.lengthscales.numpy()[0], gpflow_model.kernel.variance.numpy()])
    print(gpflow_hypers)
    print(gpflow_grads)

    np.testing.assert_allclose(newt_grads[0], gpflow_grads[0], rtol=1e-2)
    np.testing.assert_allclose(newt_grads[1], gpflow_grads[1], rtol=1e-2)


@pytest.mark.parametrize('var_f', [0.5, 1.5])
@pytest.mark.parametrize('len_f', [2.5, 5.])
@pytest.mark.parametrize('N', [30, 60])
def test_inference_step(var_f, len_f, N):
    """
    test whether newt's VI and gpflow's SVGP (Z=X) give the same posterior after one natural gradient step
    """

    x, y = build_data(N)

    newt_model = initialise_newt_model(var_f, len_f, x, y)
    gpflow_model = initialise_gpflow_model(var_f, len_f, x, y)

    lr_newton = 1.

    newt_model.inference(lr=lr_newton)  # update variational params

    data = (x, y[:, None])
    with tf.GradientTape() as tape:
        loss = -gpflow_model.elbo(data)

    variational_vars = [(gpflow_model.q_mu, gpflow_model.q_sqrt)]
    natgrad_opt = gpflow.optimizers.NaturalGradient(gamma=lr_newton)
    loss_fn = gpflow_model.training_loss_closure(data)
    natgrad_opt.minimize(loss_fn, variational_vars)

    f_mean, f_var = gpflow_model.predict_f(x)

    # print(post_mean_)
    # print(f_mean[:, 0])

    np.testing.assert_allclose(np.squeeze(newt_model.posterior_mean.value), f_mean[:, 0], rtol=5e-3)
    np.testing.assert_allclose(np.squeeze(newt_model.posterior_variance.value), f_var[:, 0], rtol=5e-3)
