import bayesnewton
import objax
from bayesnewton.utils import inv
import numpy as np
from jax.config import config
config.update("jax_enable_x64", True)
import pytest
import tensorflow as tf
import gpflow
import scipy as sp
import pickle

gpflow.config.set_default_jitter(1e-20)

train_data = pickle.load(open(f'../experiments/shutters/data/train_data_0_0.pickle', "rb"))
pred_data = pickle.load(open(f'../experiments/shutters/data/pred_data_0_0.pickle', "rb"))
pred_data = pred_data['grid']

X = train_data['X']
Y = train_data['Y']
X_test = pred_data['X']
Y_test = pred_data['Y']


def initialise_newt_model(var_f, len_f, var_y, x, y):
    r = np.unique(x[:, 1])
    print(r)
    kernel = bayesnewton.kernels.SpatioTemporalMatern52(variance=var_f, lengthscale_time=len_f, lengthscale_space=len_f, z=r)
    likelihood = bayesnewton.likelihoods.Gaussian(variance=var_y)
    model = bayesnewton.models.VariationalGP(kernel=kernel, likelihood=likelihood, X=x, Y=y)
    return model


# def init_as_cvi(kern, Z_all):
#     M = Z_all.shape[0]
#
#     Kzz = kern(Z_all, Z_all)
#
#     # def inv(K):
#     #     K_chol = sp.linalg.cholesky(K+1e-3*np.eye(M), lower=True)
#     #     return sp.linalg.cho_solve((K_chol, True), np.eye(K.shape[0]))
#
#     #manual q(u) decompositin
#     nat1 = np.zeros([M, 1])
#     nat2 = inv(Kzz)
#
#     lam1 =  np.zeros([M, 1])
#     lam2 =  1e-2*np.eye(M)
#
#     # S = inv(-2*(nat2+lam2))
#     S = inv(nat2+lam2)
#     m = S @ (lam1 + nat1)
#
#     S_chol = sp.linalg.cholesky(S+1e-8*np.eye(M), lower=True)
#     S_flattened = S_chol[np.tril_indices(M, 0)]
#
#     q_mu = m
#     q_sqrt = np.array([S_chol])
#     return q_mu, q_sqrt


def initialise_gpflow_model(var_f, len_f, var_y, x, y):
    N = x.shape[0]
    k0 = gpflow.kernels.Matern52(lengthscales=[len_f], variance=var_f, active_dims=[0], name='matern1')
    k1 = gpflow.kernels.Matern52(lengthscales=[len_f], variance=1., active_dims=[1], name='matern2')
    k = k0 * k1

    # find the m and S that correspond to the same natural parameters used by CVI
    K_xx = np.array(k(x, x))
    K_xx_inv = inv(K_xx)

    print(x.shape)

    S = inv(K_xx_inv + 1e-2 * np.eye(N))
    S_chol = np.linalg.cholesky(S)
    S_chol_init = np.array([S_chol])
    # S_chol_flattened_init = np.array(S_chol[np.tril_indices(N, 0)])

    lambda_init = np.zeros((N, 1))
    m_init = S @ lambda_init

    lik = gpflow.likelihoods.Gaussian(variance=var_y)

    # data = (x, y)
    # print(x)

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


@pytest.mark.parametrize('var_f', [1., 5.])
@pytest.mark.parametrize('len_f', [0.1, 0.025])
@pytest.mark.parametrize('var_y', [0.1, 0.3])
def test_initial_loss(var_f, len_f, var_y):
    """
    test whether newt's VI and gpflow's SVGP (Z=X) give the same initial ELBO and posterior
    """

    newt_model = initialise_newt_model(var_f, len_f, var_y, X, Y)
    gpflow_model = initialise_gpflow_model(var_f, len_f, var_y, X, Y)

    newt_model.update_posterior()
    loss_newt = newt_model.energy()
    # _, _, expected_density = newt_model.inference(newt_model)
    print(loss_newt)
    # print(expected_density)

    data = (X, Y)
    f_mean, f_var = gpflow_model.predict_f(X)
    var_exp = np.sum(gpflow_model.likelihood.variational_expectations(f_mean, f_var, Y))
    loss_gpflow = -gpflow_model.elbo(data)
    print(loss_gpflow.numpy())
    # print(var_exp)

    # print(posterior_mean - f_mean[:, 0])

    np.testing.assert_allclose(np.squeeze(newt_model.posterior_mean.value), f_mean[:, 0], rtol=1e-4)
    np.testing.assert_allclose(np.squeeze(newt_model.posterior_variance.value), f_var[:, 0], rtol=1e-4)
    np.testing.assert_almost_equal(loss_newt, loss_gpflow.numpy(), decimal=2)


@pytest.mark.parametrize('var_f', [1., 5.])
@pytest.mark.parametrize('len_f', [0.1, 0.025])
@pytest.mark.parametrize('var_y', [0.1, 0.3])
def test_gradient_step(var_f, len_f, var_y):
    """
    test whether newt's VI and gpflow's SVGP (Z=X) provide the same initial gradient step in the hyperparameters
    """

    # x, y = build_data(N)

    newt_model = initialise_newt_model(var_f, len_f, var_y, X, Y)
    gpflow_model = initialise_gpflow_model(var_f, len_f, var_y, X, Y)

    gv = objax.GradValues(newt_model.energy, newt_model.vars())

    lr_adam = 0.1
    lr_newton = 1.
    opt = objax.optimizer.Adam(newt_model.vars())

    newt_model.update_posterior()
    newt_grads, value = gv()  # , lr=lr_newton)
    loss_ = value[0]
    opt(lr_adam, newt_grads)
    newt_hypers = np.array([newt_model.kernel.temporal_lengthscale, newt_model.kernel.spatial_lengthscale,
                            newt_model.kernel.variance, newt_model.likelihood.variance])
    print(newt_hypers)
    print(newt_grads)

    adam_opt = tf.optimizers.Adam(lr_adam)
    data = (X, Y)
    with tf.GradientTape() as tape:
        loss = -gpflow_model.elbo(data)
    _vars = gpflow_model.trainable_variables
    gpflow_grads = tape.gradient(loss, _vars)

    loss_fn = gpflow_model.training_loss_closure(data)
    adam_vars = gpflow_model.trainable_variables
    adam_opt.minimize(loss_fn, adam_vars)
    #gpflow_hypers = np.array([gpflow_model.kernel.lengthscales.numpy()[0],
    #                          gpflow_model.kernel.lengthscales.numpy()[1],
    #                          gpflow_model.kernel.variance.numpy(),
    #                          gpflow_model.likelihood.variance.numpy()])
    gpflow_hypers = np.array([gpflow_model.kernel.parameters[0].numpy(),
                              gpflow_model.kernel.parameters[2].numpy(),
                              gpflow_model.kernel.parameters[1].numpy(),
                              gpflow_model.likelihood.variance.numpy()])
    print(gpflow_hypers)
    print(gpflow_grads)

    np.testing.assert_allclose(newt_grads[0], gpflow_grads[0], atol=1e-2)  # use atol since values are so small
    np.testing.assert_allclose(newt_grads[1], gpflow_grads[1], rtol=1e-2)
    np.testing.assert_allclose(newt_grads[2], gpflow_grads[2], rtol=1e-2)


# @pytest.mark.parametrize('var_f', [0.5, 1.5])
# @pytest.mark.parametrize('len_f', [0.75, 2.5])
# @pytest.mark.parametrize('var_y', [0.1, 0.5])
# def test_inference_step(var_f, len_f, var_y):
#     """
#     test whether newt's VI and gpflow's SVGP (Z=X) give the same posterior after one natural gradient step
#     """
#
#     # x, y = build_data(N)
#
#     newt_model = initialise_newt_model(var_f, len_f, var_y, X, Y)
#     gpflow_model = initialise_gpflow_model(var_f, len_f, var_y, X, Y)
#
#     lr_newton = 1.
#
#     newt_model.update_posterior()
#     newt_loss = inf(newt_model, lr=lr_newton)  # update variational params
#     newt_model.update_posterior()
#
#     data = (X, Y[:, None)
#     with tf.GradientTape() as tape:
#         loss = -gpflow_model.elbo(data)
#
#     variational_vars = [(gpflow_model.q_mu, gpflow_model.q_sqrt)]
#     natgrad_opt = gpflow.optimizers.NaturalGradient(gamma=lr_newton)
#     loss_fn = gpflow_model.training_loss_closure(data)
#     natgrad_opt.minimize(loss_fn, variational_vars)
#
#     f_mean, f_var = gpflow_model.predict_f(X)
#
#     # print(post_mean_)
#     # print(f_mean[:, 0])
#
#     np.testing.assert_allclose(np.squeeze(newt_model.posterior_mean.value), f_mean[:, 0], rtol=1e-3)
#     np.testing.assert_allclose(np.squeeze(newt_model.posterior_variance.value), f_var[:, 0], rtol=1e-3)

var_f = 1
len_f = 1
var_y = 0.1

newt_model = initialise_newt_model(var_f, len_f, var_y, X, Y)
gpflow_model = initialise_gpflow_model(var_f, len_f, var_y, X, Y)

newt_model.update_posterior()
loss_newt = newt_model.energy()
# _, _, expected_density = newt_model.inference(newt_model)
print(loss_newt)
# print(expected_density)

data = (X, Y)
f_mean, f_var = gpflow_model.predict_f(X)
var_exp = np.sum(gpflow_model.likelihood.variational_expectations(f_mean, f_var, Y))
loss_gpflow = -gpflow_model.elbo(data)
print(loss_gpflow.numpy())
# print(var_exp)

# print(posterior_mean - f_mean[:, 0])

# np.testing.assert_allclose(np.squeeze(newt_model.posterior_mean.value), f_mean[:, 0], rtol=1e-4)
# np.testing.assert_allclose(np.squeeze(newt_model.posterior_variance.value), f_var[:, 0], rtol=1e-4)
# np.testing.assert_almost_equal(loss_newt, loss_gpflow.numpy(), decimal=2)
