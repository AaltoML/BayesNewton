import bayesnewton
import objax
import numpy as np
from jax.config import config
config.update("jax_enable_x64", True)
import pytest

# TODO: ------- FIX --------


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
    model_ = bayesnewton.models.MarkovVariationalGP(kernel=kernel, likelihood=likelihood, X=x_, Y=y_)
    x_sorted = model_.X
    r_sorted = model_.R
    x_ = np.vstack([x_sorted.T, r_sorted.T]).T
    y_ = model_.Y

    model = bayesnewton.models.VariationalGP(kernel=kernel, likelihood=likelihood, X=x_, Y=y_)
    return model


def initialise_markovgp_model(var_f, len_f, var_y, x_, y_, z_):
    kernel = bayesnewton.kernels.SpatialMatern52(variance=var_f, lengthscale=len_f,
                                          z=z_, sparse=True, opt_z=False, conditional='Full')
    likelihood = bayesnewton.likelihoods.Gaussian(variance=var_y)
    model = bayesnewton.models.MarkovVariationalGP(kernel=kernel, likelihood=likelihood, X=x_, Y=y_)
    return model


@pytest.mark.parametrize('var_f', [0.5, 1.5])
@pytest.mark.parametrize('len_f', [0.75, 2.5])
@pytest.mark.parametrize('var_y', [0.1, 0.5])
@pytest.mark.parametrize('N', [8, 16])
def test_initial_loss(var_f, len_f, var_y, N):
    """
    test whether VI with newt's GP and MarkovGP give the same initial ELBO and posterior
    """

    x, Y, t, R, y = build_data(N)

    gp_model = initialise_gp_model(var_f, len_f, var_y, x, y, R[0])
    markovgp_model = initialise_markovgp_model(var_f, len_f, var_y, x, y, R[0])

    gp_model.update_posterior()
    loss_gp = gp_model.energy()
    print(loss_gp)

    markovgp_model.update_posterior()
    loss_markovgp = markovgp_model.energy()
    print(loss_markovgp)

    # print(gp_model.posterior_variance.value - markovgp_model.posterior_variance.value)

    markovgp_mean, markovgp_var = markovgp_model.predict(t, R)

    # np.testing.assert_allclose(gp_model.posterior_mean.value, markovgp_model.posterior_mean.value, rtol=1e-4)
    # np.testing.assert_allclose(gp_model.posterior_mean.value, markovgp_mean.reshape(-1, 1, 1), rtol=1e-4)
    # np.testing.assert_allclose(gp_model.posterior_variance.value, markovgp_model.posterior_variance.value, rtol=1e-4)
    # np.testing.assert_allclose(gp_model.posterior_variance.value, markovgp_var.reshape(-1, 1, 1), rtol=1e-4)
    np.testing.assert_almost_equal(loss_gp, loss_markovgp, decimal=2)


@pytest.mark.parametrize('var_f', [0.5, 1.5])
@pytest.mark.parametrize('len_f', [0.75, 2.5])
@pytest.mark.parametrize('var_y', [0.1, 0.5])
@pytest.mark.parametrize('N', [8, 16])
def test_gradient_step(var_f, len_f, var_y, N):
    """
    test whether VI with newt's GP and MarkovGP provide the same initial gradient step in the hyperparameters
    """

    x, Y, t, R, y = build_data(N)

    gp_model = initialise_gp_model(var_f, len_f, var_y, x, y, R[0])
    markovgp_model = initialise_markovgp_model(var_f, len_f, var_y, x, y, R[0])

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
    gp_hypers = np.array([gp_model.kernel.temporal_kernel.lengthscale,
                          gp_model.kernel.temporal_kernel.variance,
                          gp_model.kernel.spatial_kernel.lengthscale,
                          gp_model.likelihood.variance])
    print(gp_hypers)
    print(gp_grads)

    markovgp_model.update_posterior()
    markovgp_grads, markovgp_value = gv_markov()
    markovgp_loss_ = markovgp_value[0]
    opt_markov(lr_adam, markovgp_grads)
    markovgp_hypers = np.array([markovgp_model.kernel.temporal_kernel.lengthscale,
                                markovgp_model.kernel.temporal_kernel.variance,
                                markovgp_model.kernel.spatial_kernel.lengthscale,
                                markovgp_model.likelihood.variance])

    print(markovgp_hypers)
    print(markovgp_grads)

    np.testing.assert_allclose(gp_grads[0], markovgp_grads[0], rtol=1e-4)
    np.testing.assert_allclose(gp_grads[1], markovgp_grads[1], rtol=1e-4)
    np.testing.assert_allclose(gp_grads[2], markovgp_grads[2], rtol=1e-4)


@pytest.mark.parametrize('var_f', [0.5, 1.5])
@pytest.mark.parametrize('len_f', [0.75, 2.5])
@pytest.mark.parametrize('var_y', [0.1, 0.5])
@pytest.mark.parametrize('N', [8, 16])
def test_inference_step(var_f, len_f, var_y, N):
    """
    test whether VI with newt's GP and MarkovGP give the same posterior after one natural gradient step
    """

    x, Y, t, R, y = build_data(N)

    gp_model = initialise_gp_model(var_f, len_f, var_y, x, y, R[0])
    markovgp_model = initialise_markovgp_model(var_f, len_f, var_y, x, y, R[0])

    lr_newton = 1.

    gp_model.inference(lr=lr_newton)  # update variational params

    markovgp_model.inference(lr=lr_newton)  # update variational params

    np.testing.assert_allclose(gp_model.posterior_mean.value, markovgp_model.posterior_mean.value, rtol=1e-4)
    np.testing.assert_allclose(gp_model.posterior_variance.value, markovgp_model.posterior_variance.value, rtol=1e-4)


# N = 5
# x, Y, t, R, y = build_data(N)
#
# var_f = 0.5
# len_f = 0.75
# var_y = 0.1
#
# gp_model = initialise_gp_model(var_f, len_f, var_y, x, y, R[0])
# markovgp_model = initialise_markovgp_model(var_f, len_f, var_y, x, y, R[0])
#
# lr_newton = 1.
#
# gp_model.update_posterior()
# gp_loss = inf(gp_model, lr=lr_newton)  # update variational params
# print(gp_loss)
# gp_model.update_posterior()
#
# markovgp_model.update_posterior()
# markovgp_loss = inf(markovgp_model, lr=lr_newton)  # update variational params
# print(markovgp_loss)
# markovgp_model.update_posterior()
#
# np.testing.assert_allclose(gp_model.posterior_mean.value, markovgp_model.posterior_mean.value, rtol=1e-4)
# np.testing.assert_allclose(gp_model.posterior_variance.value, markovgp_model.posterior_variance.value, rtol=1e-4)
