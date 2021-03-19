import objax
import jax.numpy as np
from .kernels import Independent
from jax import vmap
from jax.lax import scan
from jax.ops import index, index_update
from jax.scipy.linalg import cho_factor, cho_solve
from jax.random import multivariate_normal, PRNGKey
from .utils import (
    inv,
    diag,
    solve,
    transpose,
    input_admin,
    compute_conditional_statistics,
    build_joint,
    set_z_stats,
    temporal_conditional,
    sum_natural_params_by_group,
    gaussian_expected_log_lik,
    compute_cavity
)
from .ops import (
    gaussian_conditional,
    sparse_gaussian_conditional,
    sparse_conditional_post_to_data,
    kalman_filter,
    kalman_filter_pairs,
    rauch_tung_striebel_smoother
)
import math
from jax.config import config
config.update("jax_enable_x64", True)

LOG2PI = math.log(2 * math.pi)


class Model(objax.Module):
    """
    The parent model class: initialises all the common model features and implements shared methods
    """
    def __init__(self,
                 kernel,
                 likelihood,
                 X,
                 Y,
                 func_dim=1):
        if X.ndim < 2:
            X = X[:, None]
        if Y.ndim < 2:
            Y = Y[:, None]
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.kernel = kernel
        self.likelihood = likelihood
        self.num_data = self.X.shape[0]  # number of data
        self.func_dim = func_dim  # number of latent dimensions
        self.obs_dim = Y.shape[1]  # dimensionality of the observations, Y
        self.mask = np.isnan(self.Y).reshape(Y.shape[0], Y.shape[1])
        if isinstance(self.kernel, Independent):
            pseudo_lik_size = self.func_dim  # the multi-latent case
        else:
            pseudo_lik_size = self.obs_dim
        self.pseudo_likelihood_nat1 = objax.StateVar(np.zeros([self.num_data, pseudo_lik_size, 1]))
        self.pseudo_likelihood_nat2 = objax.StateVar(1e-2 * np.tile(np.eye(pseudo_lik_size), [self.num_data, 1, 1]))
        self.pseudo_y = objax.StateVar(np.zeros([self.num_data, pseudo_lik_size, 1]))
        self.pseudo_var = objax.StateVar(1e2 * np.tile(np.eye(pseudo_lik_size), [self.num_data, 1, 1]))
        self.posterior_mean = objax.StateVar(np.zeros([self.num_data, self.func_dim, 1]))
        self.posterior_variance = objax.StateVar(np.tile(np.eye(self.func_dim), [self.num_data, 1, 1]))
        self.ind = np.arange(self.num_data)
        self.num_neighbours = np.ones(self.num_data)

    def __call__(self, X=None):
        if X is None:
            self.update_posterior()
        else:
            return self.predict(X)

    def set_pseudo_likelihood(self):
        self.pseudo_var.value = vmap(inv)(self.pseudo_likelihood_nat2.value)
        self.pseudo_y.value = self.pseudo_var.value @ self.pseudo_likelihood_nat1.value

    def prior_sample(self, num_samps=1):
        raise NotImplementedError

    def update_posterior(self):
        raise NotImplementedError

    def compute_log_lik(self, pseudo_y=None, pseudo_var=None):
        """ Compute the log likelihood of the pseudo model, i.e. the log normaliser of the approximate posterior """
        raise NotImplementedError

    def predict(self, X, R=None):
        raise NotImplementedError

    def predict_y(self, X, R=None):
        """
        predict y at new test locations X
        TODO: check non-Gaussian likelihoods
        """
        mean_f, var_f = self.predict(X, R)
        mean_f, var_f = mean_f.reshape(mean_f.shape[0], -1, 1), var_f.reshape(var_f.shape[0], -1, 1)
        mean_y, var_y = vmap(self.likelihood.predict)(mean_f, var_f)
        return np.squeeze(mean_y), np.squeeze(var_y)

    def negative_log_predictive_density(self, X, Y, R=None):
        predict_mean, predict_var = self.predict(X, R)
        if predict_mean.ndim > 1:  # multi-latent case
            pred_mean, pred_var, Y = predict_mean[..., None], diag(predict_var), Y.reshape(-1, 1)
        else:
            pred_mean, pred_var, Y = predict_mean.reshape(-1, 1, 1), predict_var.reshape(-1, 1, 1), Y.reshape(-1, 1)
        log_density = vmap(self.likelihood.log_density)(Y, pred_mean, pred_var)
        return -np.nanmean(log_density)

    def group_natural_params(self, nat1, nat2, batch_ind=None):
        if (batch_ind is not None) and (batch_ind.shape[0] != self.num_data):
            nat1 = index_update(self.pseudo_likelihood_nat1.value, index[batch_ind], nat1)
            nat2 = index_update(self.pseudo_likelihood_nat2.value, index[batch_ind], nat2)
        return nat1, nat2

    def conditional_posterior_to_data(self, batch_ind=None, post_mean=None, post_cov=None):
        if batch_ind is None:
            batch_ind = np.arange(self.num_data)
        ind = self.ind[batch_ind]
        if post_mean is None:
            post_mean = self.posterior_mean.value[ind]
        if post_cov is None:
            post_cov = self.posterior_variance.value[ind]
        return post_mean, post_cov

    def conditional_data_to_posterior(self, mean_f, cov_f):
        return mean_f, cov_f

    def expected_density_pseudo(self):
        expected_density = vmap(gaussian_expected_log_lik)(  # parallel operation
            self.pseudo_y.value,
            self.posterior_mean.value,
            self.posterior_variance.value,
            self.pseudo_var.value,
            self.mask
        )
        return np.sum(expected_density)

    def compute_kl(self):
        """
        KL divergence between the approximate posterior q(u) and the prior p(u)
        """
        # log int p(u) prod_n N(pseudo_y_n | u, pseudo_var_n) du
        log_lik_pseudo = self.compute_log_lik()
        # E_q[log N(pseudo_y_n | u, pseudo_var_n)]
        expected_density_pseudo = self.expected_density_pseudo()
        kl = expected_density_pseudo - log_lik_pseudo  # KL[approx_post || prior]
        return kl

    def compute_full_pseudo_lik(self):
        return self.pseudo_y.value, self.pseudo_var.value

    def compute_full_pseudo_nat(self, batch_ind):
        return self.pseudo_likelihood_nat1.value[batch_ind], self.pseudo_likelihood_nat2.value[batch_ind]

    def cavity_distribution(self, batch_ind=None, power=None):
        """ Compute the power EP cavity for the given data points """
        if batch_ind is None:
            batch_ind = np.arange(self.num_data)

        nat1lik_full, nat2lik_full = self.compute_full_pseudo_nat(batch_ind)

        # then compute the cavity
        cavity_mean, cavity_cov = vmap(compute_cavity, [0, 0, 0, 0, None])(
            self.posterior_mean.value[batch_ind],
            self.posterior_variance.value[batch_ind],
            nat1lik_full,
            nat2lik_full,
            power
        )
        return cavity_mean, cavity_cov


class GP(Model):
    """
    A standard (kernel-based) GP model with prior of the form
        f(t) ~ GP(0,k(t,t'))
    TODO: implement multi-latents
    """
    def __init__(self,
                 kernel,
                 likelihood,
                 X,
                 Y):
        super().__init__(kernel=kernel,
                         likelihood=likelihood,
                         X=X,
                         Y=Y)
        self.obs_ind = np.array(np.squeeze(np.where(~self.mask)[0]))  # index into observed values

    def update_posterior(self):
        """
        Compute the approximate posterior distribution using standard Gaussian identities
        """
        mean, covariance = gaussian_conditional(self.kernel,
                                                self.pseudo_y.value,
                                                self.pseudo_var.value,
                                                self.X)
        self.posterior_mean.value = mean.reshape(self.num_data, 1, 1)
        self.posterior_variance.value = np.diag(covariance).reshape(self.num_data, 1, 1)

    def compute_log_lik(self, pseudo_y=None, pseudo_var=None):
        """
        Compute the log marginal likelihood of the pseudo model, i.e. the log normaliser of the approximate posterior
        """
        dim = 1  # TODO: implement multivariate case
        # TODO: won't match MarkovGP for batching with missings or for multidim input

        X = self.X[self.obs_ind]  # only compute log lik for observed values  # TODO: check use of obs_ind (remove?)
        if pseudo_y is None:
            pseudo_y = self.pseudo_y.value
            pseudo_var = self.pseudo_var.value
        pseudo_y = pseudo_y[self.obs_ind]
        pseudo_var = pseudo_var[self.obs_ind]

        Knn = self.kernel(X, X)
        Ky = Knn + np.diag(np.squeeze(pseudo_var))  # TODO: this will break for multi-latents

        # ---- compute the marginal likelihood, i.e. the normaliser, of the pseudo model ----
        pseudo_y = diag(pseudo_y)
        Ly, low = cho_factor(Ky)
        log_lik_pseudo = (
            - 0.5 * np.sum(pseudo_y.T @ cho_solve((Ly, low), pseudo_y))
            - np.sum(np.log(np.diag(Ly)))
            - 0.5 * pseudo_y.shape[0] * dim * LOG2PI
        )

        return log_lik_pseudo

    def predict(self, X, R=None):
        """
        predict f at new test locations X
        """
        if len(X.shape) < 2:
            X = X[:, None]
        mean, covariance = gaussian_conditional(self.kernel,
                                                self.pseudo_y.value,
                                                self.pseudo_var.value,
                                                self.X,
                                                X)
        predict_mean = np.squeeze(mean)
        predict_variance = np.diag(covariance)
        return predict_mean, predict_variance

    def prior_sample(self, X=None, num_samps=1, key=0):
        if X is None:
            X = self.X
        N = X.shape[0]
        m = np.zeros(N)
        K = self.kernel(X, X) + 1e-12 * np.eye(N)
        s = multivariate_normal(PRNGKey(key), m, K, shape=[num_samps])
        return s.T


class SparseGP(GP):
    """
    A standard (kernel-based) GP model with prior of the form
        f(t) ~ GP(0,k(t,t'))
    :param opt_z: flag whether to optimise the inducing inputs Z
    TODO: write test comparing to gpflow
    TODO: implement multi-latents
    """
    def __init__(self,
                 kernel,
                 likelihood,
                 X,
                 Y,
                 Z,
                 opt_z=False):
        super().__init__(kernel=kernel,
                         likelihood=likelihood,
                         X=X,
                         Y=Y)
        if Z.ndim < 2:
            Z = Z[:, None]
        if opt_z:
            self.Z = objax.TrainVar(Z)
        else:
            self.Z = objax.StateVar(Z)
        self.num_inducing = Z.shape[0]
        self.posterior_mean = objax.StateVar(np.zeros([self.num_inducing, self.func_dim, 1]))
        self.posterior_variance = objax.StateVar(np.tile(np.eye(self.func_dim), [self.num_inducing, 1, 1]))
        self.posterior_covariance = objax.StateVar(np.eye(self.num_inducing))

    def update_posterior(self):
        """
        Compute the approximate posterior distribution using standard Gaussian identities
        """
        mean, covariance = sparse_gaussian_conditional(self.kernel,
                                                       self.pseudo_likelihood_nat1.value,
                                                       self.pseudo_likelihood_nat2.value,
                                                       self.X,
                                                       self.Z.value)
        self.posterior_mean.value = mean.reshape(self.num_inducing, 1, 1)
        self.posterior_variance.value = np.diag(covariance).reshape(self.num_inducing, 1, 1)
        self.posterior_covariance.value = covariance.reshape(self.num_inducing, self.num_inducing)

    def compute_full_pseudo_lik(self):
        """ The pseudo-likelihoods are currently stored as N Gaussians in f - convert to M Gaussian in u """
        Kuf = self.kernel(self.Z.value, self.X[self.obs_ind])  # only compute log lik for observed values
        Kuu = self.kernel(self.Z.value, self.Z.value)
        Wuf = solve(Kuu, Kuf)  # conditional mapping, Kuu^-1 Kuf

        # TODO: more efficient way to do this?
        nat1lik_full = Wuf @ np.squeeze(self.pseudo_likelihood_nat1.value[self.obs_ind], axis=-1)
        nat2lik_full = Wuf @ np.diag(np.squeeze(self.pseudo_likelihood_nat2.value[self.obs_ind])) @ transpose(Wuf)
        pseudo_var_full = inv(nat2lik_full + 1e-12 * np.eye(Kuu.shape[0]))
        pseudo_y_full = pseudo_var_full @ nat1lik_full
        return pseudo_y_full, pseudo_var_full

    def compute_kl(self):
        """
        KL divergence between the approximate posterior q(u) and the prior p(u)
        """
        pseudo_y_full, pseudo_var_full = self.compute_full_pseudo_lik()

        # ---- compute the log marginal likelihood, i.e. the normaliser, of the pseudo model ----
        # log int p(u) prod_n N(pseudo_y_n | u, pseudo_var_n) du
        log_lik_pseudo = self.compute_log_lik(pseudo_y_full, pseudo_var_full)

        # E_q[log N(pseudo_y_n | u, pseudo_var_n)]
        expected_density_pseudo = gaussian_expected_log_lik(  # this term does not depend on the prior, use stored q(u)
            pseudo_y_full,
            np.squeeze(self.posterior_mean.value, axis=-1),
            self.posterior_covariance.value,
            pseudo_var_full
        )

        kl = expected_density_pseudo - log_lik_pseudo  # KL[approx_post || prior]
        return kl

    def compute_log_lik(self, pseudo_y=None, pseudo_var=None):
        """ log int p(u) prod_n N(pseudo_y_n | u, pseudo_var_n) du """
        dim = 1  # TODO: implement multivariate case
        Kuu = self.kernel(self.Z.value, self.Z.value)

        Ky = Kuu + pseudo_var
        Ly, low = cho_factor(Ky)
        log_lik_pseudo = (  # this term depends on the prior
            - 0.5 * np.sum(pseudo_y.T @ cho_solve((Ly, low), pseudo_y))
            - np.sum(np.log(np.diag(Ly)))
            - 0.5 * pseudo_y.shape[0] * dim * LOG2PI
        )
        return log_lik_pseudo

    def predict(self, X, R=None):
        """
        predict at new test locations X
        """
        if len(X.shape) < 2:
            X = X[:, None]
        self.update_posterior()
        mean, covariance = sparse_conditional_post_to_data(self.kernel,
                                                           self.posterior_mean.value,
                                                           self.posterior_covariance.value,
                                                           X,
                                                           self.Z.value)
        predict_mean = np.squeeze(mean)
        predict_variance = np.diag(covariance)
        return predict_mean, predict_variance

    def conditional_posterior_to_data(self, batch_ind=None, post_mean=None, post_cov=None):
        """
        compute
        q(f) = int p(f | u) q(u) du
        where
        q(u) = N(u | post_mean, post_cov)
        """
        if batch_ind is None:
            batch_ind = np.arange(self.num_data)
        if post_mean is None:
            post_mean = self.posterior_mean.value
        if post_cov is None:
            post_cov = self.posterior_covariance.value

        mean_f, cov_f = sparse_conditional_post_to_data(self.kernel,
                                                        post_mean,
                                                        post_cov,
                                                        self.X[batch_ind],
                                                        self.Z.value)

        Nbatch = batch_ind.shape[0]
        return mean_f.reshape(Nbatch, 1, 1), np.diag(cov_f).reshape(Nbatch, 1, 1)

    def prior_sample(self, X=None, num_samps=1, key=0):
        # TODO: implement using objax.random
        raise NotImplementedError


class MarkovGP(Model):
    """
    The stochastic differential equation (SDE) form of a Gaussian process (GP) model.
    Implements methods for inference and learning using state space methods, i.e. Kalman filtering and smoothing.
    Constructs a linear time-invariant (LTI) stochastic differential equation (SDE) of the following form:
        dx(t)/dt = F x(t) + L w(t)
              yₙ ~ p(yₙ | f(t_n)=H x(t_n))
    where w(t) is a white noise process and where the state x(t) is Gaussian distributed with initial
    state distribution x(t)~𝓝(0,Pinf).
    """
    def __init__(self,
                 kernel,
                 likelihood,
                 X,
                 Y,
                 R=None):
        (X, Y, self.R, self.dt) = input_admin(X, Y, R)
        H = kernel.measurement_model()
        func_dim = H.shape[0]  # number of latent dimensions
        super().__init__(kernel=kernel,
                         likelihood=likelihood,
                         X=X,
                         Y=Y,
                         func_dim=func_dim)
        self.state_dim = self.kernel.stationary_covariance().shape[0]
        self.minf = np.zeros([self.state_dim, 1])  # stationary state mean
        self.spatio_temporal = np.any(~np.isnan(self.R))

    @staticmethod
    def filter(*args, **kwargs):
        return kalman_filter(*args, **kwargs)

    @staticmethod
    def smoother(*args, **kwargs):
        return rauch_tung_striebel_smoother(*args, **kwargs)

    @staticmethod
    def temporal_conditional(*args, **kwargs):
        return temporal_conditional(*args, **kwargs)

    def compute_full_pseudo_nat(self, batch_ind):
        if self.spatio_temporal:  # spatio-temporal case
            B, C = self.kernel.spatial_conditional(self.X[batch_ind], self.R[batch_ind])
            nat1lik_full = transpose(B) @ self.pseudo_likelihood_nat1.value[batch_ind]
            nat2lik_full = transpose(B) @ self.pseudo_likelihood_nat2.value[batch_ind] @ B
            return nat1lik_full, nat2lik_full
        else:  # temporal case
            return self.pseudo_likelihood_nat1.value[batch_ind], self.pseudo_likelihood_nat2.value[batch_ind]

    def compute_full_pseudo_lik(self):
        # TODO: running this 3 times per training loop is wasteful - store in memory?
        if self.spatio_temporal:  # spatio-temporal case
            B, C = self.kernel.spatial_conditional(self.X, self.R)
            # TODO: more efficient way to do this?
            nat1lik_full = transpose(B) @ self.pseudo_likelihood_nat1.value
            nat2lik_full = transpose(B) @ self.pseudo_likelihood_nat2.value @ B
            pseudo_var_full = vmap(inv)(nat2lik_full + 1e-12 * np.eye(nat2lik_full.shape[1]))  # <---------- bottleneck
            pseudo_y_full = pseudo_var_full @ nat1lik_full
            return pseudo_y_full, pseudo_var_full
        else:  # temporal case
            return self.pseudo_y.value, self.pseudo_var.value

    def update_posterior(self):
        """
        Compute the posterior via filtering and smoothing
        """
        pseudo_y, pseudo_var = self.compute_full_pseudo_lik()
        mask = self.mask
        if mask.shape[1] != pseudo_y.shape[1]:  # TODO: store in memory?
            mask = np.tile(self.mask, [1, pseudo_y.shape[1]])
        log_lik, (filter_mean, filter_cov) = self.filter(self.dt,
                                                         self.kernel,
                                                         pseudo_y,
                                                         pseudo_var,
                                                         mask)
        dt = np.concatenate([self.dt[1:], np.array([0.0])], axis=0)
        smoother_mean, smoother_cov, _ = self.smoother(dt,
                                                       self.kernel,
                                                       filter_mean,
                                                       filter_cov)
        self.posterior_mean.value, self.posterior_variance.value = smoother_mean, smoother_cov

    def compute_kl(self):
        """
        KL[q()|p()]
        """
        pseudo_y, pseudo_var = self.compute_full_pseudo_lik()
        log_lik_pseudo = self.compute_log_lik(pseudo_y, pseudo_var)

        mask = self.mask
        if mask.shape[1] != pseudo_y.shape[1]:  # TODO: store in memory?
            mask = np.tile(self.mask, [1, pseudo_y.shape[1]])

        expected_density_pseudo = vmap(gaussian_expected_log_lik)(  # parallel operation
            pseudo_y,
            self.posterior_mean.value,
            self.posterior_variance.value,
            pseudo_var,
            mask
        )

        kl = np.sum(expected_density_pseudo) - log_lik_pseudo  # KL[approx_post || prior]
        return kl

    def compute_log_lik(self, pseudo_y=None, pseudo_var=None):
        """
        int p(f) N(pseudo_y | f, pseudo_var) df
        """
        if pseudo_y is None:
            pseudo_y, pseudo_var = self.compute_full_pseudo_lik()

        mask = self.mask
        if mask.shape[1] != pseudo_y.shape[1]:  # TODO: store in memory?
            mask = np.tile(self.mask, [1, pseudo_y.shape[1]])

        log_lik_pseudo, (_, _) = self.filter(
            self.dt,
            self.kernel,
            pseudo_y,
            pseudo_var,
            mask
        )
        return log_lik_pseudo

    def conditional_posterior_to_data(self, batch_ind=None, post_mean=None, post_cov=None):
        """
        compute
        q(f) = int p(f | u) q(u) du = N(f | B post_mean, B post_cov B' + C)
        where
        q(u) = N(u | post_mean, post_cov)
        p(f | u) = N(f | Bu, C)
        """
        if batch_ind is None:
            batch_ind = np.arange(self.num_data)
        if post_mean is None:
            post_mean = self.posterior_mean.value[batch_ind]
        if post_cov is None:
            post_cov = self.posterior_variance.value[batch_ind]

        if self.spatio_temporal:
            B, C = self.kernel.spatial_conditional(self.X[batch_ind], self.R[batch_ind])
            mean_f = B @ post_mean
            cov_f = B @ post_cov @ transpose(B) + C
            return mean_f, cov_f
        else:
            return post_mean, post_cov

    def predict(self, X, R=None):
        """
        predict at new test locations X
        """
        if len(X.shape) < 2:
            X = X[:, None]
        if R is None:
            R = X[:, 1:]
        X = X[:, :1]  # take only the temporal component

        pseudo_y, pseudo_var = self.compute_full_pseudo_lik()
        _, (filter_mean, filter_cov) = self.filter(self.dt,
                                                   self.kernel,
                                                   pseudo_y,
                                                   pseudo_var)
        dt = np.concatenate([self.dt[1:], np.array([0.0])], axis=0)
        smoother_mean, smoother_cov, gain = self.smoother(dt,
                                                          self.kernel,
                                                          filter_mean,
                                                          filter_cov,
                                                          return_full=True)

        # add dummy states at either edge
        inf = 1e10 * np.ones_like(self.X[0, :1])
        X_aug = np.block([[-inf], [self.X[:, :1]], [inf]])

        # predict the state distribution at the test time steps:
        state_mean, state_cov = self.temporal_conditional(X_aug, X, smoother_mean, smoother_cov, gain, self.kernel)
        # extract function values from the state:
        H = self.kernel.measurement_model()
        if self.spatio_temporal:
            # TODO: if R is fixed, only compute B, C once
            B, C = self.kernel.spatial_conditional(X, R)
            W = B @ H
            test_mean = W @ state_mean
            test_var = W @ state_cov @ transpose(W) + C
        else:
            test_mean, test_var = H @ state_mean, H @ state_cov @ transpose(H)

        if np.squeeze(test_var).ndim > 2:  # deal with spatio-temporal case (discard spatial covariance)
            test_var = diag(np.squeeze(test_var))
        return np.squeeze(test_mean), np.squeeze(test_var)

    def prior_sample(self, X=None, num_samps=1, key=0):
        # TODO: implement using objax.random
        raise NotImplementedError

    def filter_energy(self):
        pseudo_y, pseudo_var = self.compute_full_pseudo_lik()
        _, (filter_mean, filter_cov) = self.filter(self.dt,
                                                   self.kernel,
                                                   pseudo_y,
                                                   pseudo_var,
                                                   return_predict=True)
        H = self.kernel.measurement_model()
        mean = H @ filter_mean
        var = H @ filter_cov @ transpose(H)
        filter_energy = -np.sum(vmap(self.likelihood.log_density)(self.Y, mean, var))
        return filter_energy


class SparseMarkovGP(MarkovGP):
    """
    A sparse Markovian GP.
    TODO: implement version with non-tied sites
    """
    def __init__(self,
                 kernel,
                 likelihood,
                 X,
                 Y,
                 R=None,
                 Z=None):
        super().__init__(kernel=kernel,
                         likelihood=likelihood,
                         X=X,
                         Y=Y,
                         R=R)
        if Z is None:
            Z = self.X
        else:
            if Z.ndim < 2:
                Z = Z[:, None]
            Z = np.sort(Z, axis=0)
        inf = np.array([[1e10]])
        self.Z = objax.StateVar(np.concatenate([-inf, Z, inf], axis=0))
        self.dz = np.array(np.diff(self.Z.value[:, 0]))
        self.num_transitions = self.dz.shape[0]
        zeros = np.zeros([self.num_transitions, 2 * self.state_dim, 1])
        eyes = np.tile(np.eye(2 * self.state_dim), [self.num_transitions, 1, 1])

        # nat2 = 1e-8 * eyes

        # initialise to match MarkovGP / GP on first step (when Z=X):
        nat2 = index_update(1e-8 * eyes, index[:-1, self.state_dim, self.state_dim], 1e-2)

        # initialise to match old implementation:
        # nat2 = (1 / 99) * eyes

        self.pseudo_likelihood_nat1 = objax.StateVar(zeros)
        self.pseudo_likelihood_nat2 = objax.StateVar(nat2)
        self.pseudo_y = objax.StateVar(zeros)
        self.pseudo_var = objax.StateVar(vmap(inv)(nat2))
        self.posterior_mean = objax.StateVar(zeros)
        self.posterior_variance = objax.StateVar(eyes)
        self.mask = None
        self.conditional_mean = None
        # TODO: if training Z this needs to be done at every training step (as well as sorting and computing dz)
        self.ind, self.num_neighbours = set_z_stats(self.X, self.Z.value)

    @staticmethod
    def filter(*args, **kwargs):
        return kalman_filter_pairs(*args, **kwargs)

    @staticmethod
    def smoother(*args, **kwargs):
        return rauch_tung_striebel_smoother(*args, **kwargs)

    def compute_full_pseudo_lik(self):
        return self.pseudo_y.value, self.pseudo_var.value

    def update_posterior(self):
        """
        Compute the posterior via filtering and smoothing
        """
        log_lik, (filter_mean, filter_cov) = self.filter(self.dz,
                                                         self.kernel,
                                                         self.pseudo_y.value,
                                                         self.pseudo_var.value)
        dz = self.dz[1:]
        smoother_mean, smoother_cov, gain = self.smoother(dz,
                                                          self.kernel,
                                                          filter_mean,
                                                          filter_cov,
                                                          return_full=True)

        minf, Pinf = self.minf[None, ...], self.kernel.stationary_covariance()[None, ...]
        mean_aug = np.concatenate([minf, smoother_mean, minf])
        cov_aug = np.concatenate([Pinf, smoother_cov, Pinf])
        gain = np.concatenate([np.zeros_like(gain[:1]), gain])
        # construct the joint distribution between neighbouring pairs of states
        post_mean, post_cov = vmap(build_joint, [0, None, None, None])(
            np.arange(self.num_transitions), mean_aug, cov_aug, gain
        )

        self.posterior_mean.value, self.posterior_variance.value = post_mean, post_cov

    def compute_log_lik(self, pseudo_y=None, pseudo_var=None):
        """
        Compute the log marginal likelihood of the pseudo model, i.e. the log normaliser of the approximate posterior
        """
        log_lik, (_, _) = self.filter(self.dz,
                                      self.kernel,
                                      self.pseudo_y.value,
                                      self.pseudo_var.value)
        return log_lik

    def compute_kl(self):
        """
        KL divergence between the approximate posterior q(u) and the prior p(u)
        """
        # log int p(u) prod_n N(pseudo_y_n | u, pseudo_var_n) du
        log_lik_pseudo = self.compute_log_lik()
        # E_q[log N(pseudo_y_n | u, pseudo_var_n)]
        expected_density_pseudo = self.expected_density_pseudo()
        kl = expected_density_pseudo - log_lik_pseudo  # KL[approx_post || prior]
        return kl

    def predict(self, X, R=None):
        """
        predict at new test locations X
        """
        if len(X.shape) < 2:
            X = X[:, None]
        if R is None:
            R = X[:, 1:]
        X = X[:, :1]  # take only the temporal component

        _, (filter_mean, filter_cov) = self.filter(self.dz,
                                                   self.kernel,
                                                   self.pseudo_y.value,
                                                   self.pseudo_var.value)
        dz = self.dz[1:]
        smoother_mean, smoother_cov, gain = self.smoother(dz,
                                                          self.kernel,
                                                          filter_mean,
                                                          filter_cov,
                                                          return_full=True)

        # predict the state distribution at the test time steps
        state_mean, state_cov = self.temporal_conditional(self.Z.value, X, smoother_mean, smoother_cov,
                                                          gain, self.kernel)
        # extract function values from the state:
        H = self.kernel.measurement_model()
        if self.spatio_temporal:
            # TODO: if R is fixed, only compute B, C once
            B, C = self.kernel.spatial_conditional(X, R)
            W = B @ H
            test_mean = W @ state_mean
            test_var = W @ state_cov @ transpose(W) + C
        else:
            test_mean, test_var = H @ state_mean, H @ state_cov @ transpose(H)

        if np.squeeze(test_var).ndim > 2:  # deal with spatio-temporal case (discard spatial covariance)
            test_var = diag(np.squeeze(test_var))
        return np.squeeze(test_mean), np.squeeze(test_var)

    def conditional_posterior_to_data(self, batch_ind=None, post_mean=None, post_cov=None):
        """
        compute
        q(f) = int p(f | u) q(u) du
        where
        q(u) = N(u | post_mean, post_cov)
        """
        if batch_ind is None:
            batch_ind = np.arange(self.num_data)
        if post_mean is None:
            post_mean = self.posterior_mean.value
        if post_cov is None:
            post_cov = self.posterior_variance.value
        ind = self.ind[batch_ind]
        post_mean, post_cov = post_mean[ind], post_cov[ind]

        P, T = vmap(compute_conditional_statistics, [0, None, None, 0])(
            self.X[batch_ind, :1], self.Z.value, self.kernel, ind
        )

        H = self.kernel.measurement_model()
        if self.spatio_temporal:
            B, C = self.kernel.spatial_conditional(self.X[batch_ind], self.R[batch_ind])
            BH = B @ H
            self.conditional_mean = BH @ P  # W
            conditional_cov = BH @ T @ transpose(BH) + C  # nu
        else:
            self.conditional_mean = H @ P  # W
            conditional_cov = H @ T @ transpose(H)  # nu

        mean_f = self.conditional_mean @ post_mean
        cov_f = self.conditional_mean @ post_cov @ transpose(self.conditional_mean) + conditional_cov

        return mean_f, cov_f

    def conditional_data_to_posterior(self, mean_f, cov_f):
        """
        conditional_posterior_to_data() must be run first so that self.conditional_mean is set
        """
        mean_q = transpose(self.conditional_mean) @ mean_f
        cov_q = transpose(self.conditional_mean) @ cov_f @ self.conditional_mean
        return mean_q, cov_q

    def group_natural_params(self, nat1_n, nat2_n, batch_ind=None):

        if batch_ind is None:
            ind = self.ind
        else:
            ind = self.ind[batch_ind]

        old_nat1 = self.pseudo_likelihood_nat1.value
        old_nat2 = self.pseudo_likelihood_nat2.value

        (new_nat1, new_nat2, counter), _ = scan(f=sum_natural_params_by_group,
                                                init=(np.zeros_like(old_nat1),
                                                      np.zeros_like(old_nat2),
                                                      np.zeros(old_nat1.shape[0])),
                                                xs=(ind, nat1_n, nat2_n))

        num_neighbours = np.maximum(self.num_neighbours, 1).reshape(-1, 1, 1)
        counter = counter.reshape(-1, 1, 1)
        nat1 = new_nat1 + (1. - counter / num_neighbours) * old_nat1
        nat2 = new_nat2 + (1. - counter / num_neighbours) * old_nat2

        nat2 += 1e-8 * np.eye(nat2.shape[1])  # prevent zeros

        return nat1, nat2

    def cavity_distribution(self, batch_ind=None, power=None):
        """ Compute the power EP cavity for the given data points """
        fraction = power / np.maximum(self.num_neighbours, 1)
        cavity_mean, cavity_cov = vmap(compute_cavity)(
            self.posterior_mean.value,
            self.posterior_variance.value,
            self.pseudo_likelihood_nat1.value,
            self.pseudo_likelihood_nat2.value,
            fraction
        )
        return cavity_mean, cavity_cov

    def prior_sample(self, X=None, num_samps=1, key=0):
        # TODO: implement using objax.random
        raise NotImplementedError
