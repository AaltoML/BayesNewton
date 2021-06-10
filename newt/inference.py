import objax
import jax.numpy as np
from jax import vmap
from jax.ops import index_update, index
from .utils import (
    diag,
    transpose,
    inv_vmap,
    solve,
    ensure_diagonal_positive_precision,
    mvn_logpdf,
    pep_constant
)
from .likelihoods import Likelihood
from .models import SparseGP
from .basemodels import GaussianDistribution
import math
import abc

LOG2PI = math.log(2 * math.pi)


def newton_update(mean, jacobian, hessian):
    """
    Applies one step of Newton's method to update the pseudo_likelihood parameters.
    Note that this is the natural parameter form of a Newton step.
    """

    # deal with missing data
    hessian = np.where(np.isnan(hessian), -1e-6, hessian)
    jacobian = np.where(np.isnan(jacobian), hessian @ mean, jacobian)

    # Newton update
    pseudo_likelihood_nat1 = (
        jacobian - hessian @ mean
    )
    pseudo_likelihood_nat2 = (
        -hessian
    )

    return pseudo_likelihood_nat1, pseudo_likelihood_nat2


@vmap
def mvn_logpdf_(*args, **kwargs):
    return mvn_logpdf(*args, **kwargs)


class InferenceMixin(abc.ABC):
    """
    The approximate inference class. To be used as a Mixin, to add inference functionality to the model class.
    Each approximate inference scheme implements an 'update_()' method which is called during
    inference in order to update the local likelihood approximation (the sites).
    TODO: improve code sharing between classes
    TODO: re-derive and re-implement QuasiNewton methods
    TODO: move as much of the generic functionality as possible from the base model class to this class.
    """
    num_data: float
    Y: np.DeviceArray
    ind: np.DeviceArray
    pseudo_likelihood: GaussianDistribution
    posterior_mean: objax.StateVar
    posterior_var: objax.StateVar
    update_posterior: classmethod
    group_natural_params: classmethod
    set_pseudo_likelihood: classmethod
    conditional_posterior_to_data: classmethod
    conditional_data_to_posterior: classmethod
    likelihood: Likelihood

    def inference(self, lr=1., batch_ind=None, **kwargs):

        if (batch_ind is None) or (batch_ind.shape[0] == self.num_data):
            batch_ind = None

        self.update_posterior()  # make sure the posterior is up to date

        # use the chosen inference method (VI, EP, ...) to compute the necessary terms for the parameter update
        mean, jacobian, hessian = self.update_variational_params(batch_ind, lr, **kwargs)
        # ---- Newton update ----
        nat1_n, nat2_n = newton_update(mean, jacobian, hessian)
        # -----------------------
        nat1, nat2 = self.group_natural_params(nat1_n, nat2_n, batch_ind)  # only required for SparseMarkov models

        # ---- update the model variational parameters ----
        self.pseudo_likelihood.update_nat_params(
            nat1=(1 - lr) * self.pseudo_likelihood.nat1 + lr * nat1,
            nat2=(1 - lr) * self.pseudo_likelihood.nat2 + lr * nat2
        )

        self.update_posterior()  # recompute posterior with new params

        return mean, jacobian, hessian  # output state to be used in linesearch methods

    def update_variational_params(self, batch_ind=None, lr=1., **kwargs):
        raise NotImplementedError

    def energy(self, batch_ind=None, **kwargs):
        raise NotImplementedError


class Laplace(InferenceMixin):
    """
    """
    compute_kl: classmethod

    def update_variational_params(self, batch_ind=None, lr=1., **kwargs):
        """
        """
        if batch_ind is None:
            batch_ind = np.arange(self.num_data)

        mean_f, _ = self.conditional_posterior_to_data(batch_ind)

        # Laplace approximates the expected density with a point estimate at the posterior mean: log p(y|f=m)
        log_lik, jacobian, hessian = vmap(self.likelihood.log_likelihood_gradients)(  # parallel
            self.Y[batch_ind],
            mean_f
        )

        hessian = -ensure_diagonal_positive_precision(-hessian)  # manual fix to avoid non-PSD precision

        jacobian, hessian = self.conditional_data_to_posterior(jacobian[..., None], hessian)

        if mean_f.shape[1] == jacobian.shape[1]:
            return mean_f, jacobian, hessian
        else:
            ind = self.ind[batch_ind]
            return self.posterior_mean.value[ind], jacobian, hessian  # sparse Markov case

    def energy(self, batch_ind=None, **kwargs):
        """
        TODO: implement correct Laplace energy
        """
        if batch_ind is None:
            batch_ind = np.arange(self.num_data)
            scale = 1
        else:
            scale = self.num_data / batch_ind.shape[0]

        mean_f, _ = self.conditional_posterior_to_data(batch_ind)

        # Laplace approximates the expected density with a point estimate at the posterior mean: log p(y|f=m)
        log_lik, _, _ = vmap(self.likelihood.log_likelihood_gradients)(  # parallel
            self.Y[batch_ind],
            mean_f
        )

        KL = self.compute_kl()  # KL[q(f)|p(f)]
        laplace_energy = -(  # Laplace approximation to the negative log marginal likelihood
            scale * np.nansum(log_lik)  # nansum accounts for missing data
            - KL
        )

        return laplace_energy


class VariationalInference(InferenceMixin):
    """
    Natural gradient VI (using the conjugate-computation VI approach)
    Refs:
        Khan & Lin 2017 "Conugate-computation variational inference - converting inference
                         in non-conjugate models in to inference in conjugate models"
        Chang, Wilkinson, Khan & Solin 2020 "Fast variational learning in state space Gaussian process models"
    """
    compute_kl: classmethod

    def update_variational_params(self, batch_ind=None, lr=1., cubature=None, **kwargs):
        """
        """
        if batch_ind is None:
            batch_ind = np.arange(self.num_data)

        mean_f, cov_f = self.conditional_posterior_to_data(batch_ind)

        # VI expected density is E_q[log p(y|f)]
        expected_density, dE_dm, d2E_dm2 = vmap(self.likelihood.variational_expectation, (0, 0, 0, None))(
            self.Y[batch_ind],
            mean_f,
            cov_f,
            cubature
        )

        d2E_dm2 = -ensure_diagonal_positive_precision(-d2E_dm2)  # manual fix to avoid non-PSD precision

        jacobian, hessian = self.conditional_data_to_posterior(dE_dm, d2E_dm2)

        if mean_f.shape[1] == jacobian.shape[1]:
            return mean_f, jacobian, hessian
        else:
            ind = self.ind[batch_ind]
            return self.posterior_mean.value[ind], jacobian, hessian  # sparse Markov case

    def energy(self, batch_ind=None, cubature=None, **kwargs):
        """
        """
        if batch_ind is None:
            batch_ind = np.arange(self.num_data)
            scale = 1
        else:
            scale = self.num_data / batch_ind.shape[0]

        mean_f, cov_f = self.conditional_posterior_to_data(batch_ind)

        # VI expected density is E_q[log p(y|f)]
        expected_density, _, _ = vmap(self.likelihood.variational_expectation, (0, 0, 0, None))(
            self.Y[batch_ind],
            mean_f,
            cov_f,
            cubature
        )

        KL = self.compute_kl()  # KL[q(f)|p(f)]
        variational_free_energy = -(  # the variational free energy, i.e., the negative ELBO
            scale * np.nansum(expected_density)  # nansum accounts for missing data
            - KL
        )

        return variational_free_energy


class ExpectationPropagation(InferenceMixin):
    """
    Expectation propagation (EP)
    """
    cavity_distribution: classmethod
    compute_full_pseudo_lik: classmethod
    mask_y: np.DeviceArray
    mask_pseudo_y: np.DeviceArray

    def update_variational_params(self, batch_ind=None, lr=1., cubature=None, power=1.):
        """
        """
        if batch_ind is None:
            batch_ind = np.arange(self.num_data)

        # compute the cavity distribution
        cavity_mean, cavity_cov = self.cavity_distribution(batch_ind, power)
        cav_mean_f, cav_cov_f = self.conditional_posterior_to_data(batch_ind, cavity_mean, cavity_cov)

        # calculate log marginal likelihood and the new sites via moment matching:
        # EP expected density is log E_q[p(y|f)]
        lZ, dlZ, d2lZ = vmap(self.likelihood.moment_match, (0, 0, 0, None, None))(
            self.Y[batch_ind],
            cav_mean_f,
            cav_cov_f,
            power,
            cubature
        )

        cav_prec = inv_vmap(cav_cov_f)
        scale_factor = cav_prec @ inv_vmap(d2lZ + cav_prec) / power  # this form guarantees symmetry

        dlZ = scale_factor @ dlZ
        d2lZ = scale_factor @ d2lZ
        if self.mask_pseudo_y is not None:
            # apply mask
            mask = self.mask_pseudo_y[batch_ind][..., None]
            dlZ = np.where(mask, np.nan, dlZ)
            d2lZ_masked = np.where(mask + transpose(mask), 0., d2lZ)  # ensure masked entries are independent
            d2lZ = np.where(diag(mask)[..., None], np.nan, d2lZ_masked)  # ensure masked entries return log like of 0

        d2lZ = -ensure_diagonal_positive_precision(-d2lZ)  # manual fix to avoid non-PSD precision

        jacobian, hessian = self.conditional_data_to_posterior(dlZ, d2lZ)

        if cav_mean_f.shape[1] == jacobian.shape[1]:
            return cav_mean_f, jacobian, hessian
        else:
            ind = self.ind[batch_ind]
            return cavity_mean[ind], jacobian, hessian  # sparse Markov case

    def energy(self, batch_ind=None, cubature=None, power=1.):
        """
        TODO: the energy is incorrect for SparseGP
        """
        if batch_ind is None:
            batch_ind = np.arange(self.num_data)
            scale = 1
        else:
            scale = self.num_data / batch_ind.shape[0]

        # compute the cavity distribution
        cavity_mean, cavity_cov = self.cavity_distribution(None, power)  # TODO: check batch_ind is not required
        cav_mean_f, cav_cov_f = self.conditional_posterior_to_data(None, cavity_mean, cavity_cov)

        # calculate log marginal likelihood and the new sites via moment matching:
        # EP expected density is log E_q[p(y|f)]
        lZ, _, _ = vmap(self.likelihood.moment_match, (0, 0, 0, None, None))(
            self.Y[batch_ind],
            cav_mean_f[batch_ind],
            cav_cov_f[batch_ind],
            power,
            cubature
        )

        if self.mask_y is not None:
            if np.squeeze(self.mask_y[batch_ind]).ndim != np.squeeze(lZ).ndim:
                raise NotImplementedError('masking in spatio-temporal models not implemented for EP')
            lZ = np.where(np.squeeze(self.mask_y[batch_ind]), 0., np.squeeze(lZ))  # apply mask

        pseudo_y, pseudo_var = self.compute_full_pseudo_lik()
        lZ_pseudo = mvn_logpdf_(
            pseudo_y,
            cavity_mean,
            pseudo_var / power + cavity_cov,
            self.mask_pseudo_y
        )
        constant = vmap(pep_constant, [0, None, 0])(pseudo_var, power, self.mask_pseudo_y)  # PEP constant
        lZ_pseudo += constant

        if isinstance(self, SparseGP):
            pseudo_y, pseudo_var = self.compute_global_pseudo_lik()
            # TODO: check derivation for SparseGP. May take different form
            # lZ_post = np.sum(vmap(self.compute_log_lik)(pseudo_y, pseudo_var))
        # else:
        lZ_post = self.compute_log_lik(pseudo_y, pseudo_var)

        ep_energy = -(
            lZ_post
            + 1 / power * (scale * np.nansum(lZ) - np.nansum(lZ_pseudo))
        )

        return ep_energy


class PosteriorLinearisation(InferenceMixin):
    """
    An iterated smoothing algorithm based on statistical linear regression (SLR).
    This method linearises the likelihood model in the region described by the posterior.
    """
    # TODO: remove these when possible
    cavity_distribution: classmethod
    compute_full_pseudo_lik: classmethod
    mask_y: np.DeviceArray
    mask_pseudo_y: np.DeviceArray

    def update_variational_params(self, batch_ind=None, lr=1., cubature=None, **kwargs):
        """
        """
        if batch_ind is None:
            batch_ind = np.arange(self.num_data)

        mean_f, cov_f = self.conditional_posterior_to_data(batch_ind)

        # PL expected density is mu=E_q[E(y|f)]
        mu, omega, d_mu, _ = vmap(self.likelihood.statistical_linear_regression, (0, 0, None))(
            mean_f,
            cov_f,
            cubature
        )
        residual = self.Y[batch_ind].reshape(mu.shape) - mu
        mask = np.isnan(residual)
        residual = np.where(mask, 0., residual)

        dmu_omega = transpose(solve(omega, d_mu))  # d_mu^T inv(omega)
        jacobian = dmu_omega @ residual
        hessian = -dmu_omega @ d_mu

        hessian = -ensure_diagonal_positive_precision(-hessian)  # manual fix to avoid non-PSD precision

        # deal with missing data
        jacobian = np.where(mask, np.nan, jacobian)
        hessian = np.where(mask * np.eye(mask.shape[1]), np.nan, hessian)

        jacobian, hessian = self.conditional_data_to_posterior(jacobian, hessian)

        if mean_f.shape[1] == jacobian.shape[1]:
            return mean_f, jacobian, hessian
        else:
            ind = self.ind[batch_ind]
            return self.posterior_mean.value[ind], jacobian, hessian  # sparse Markov case

    def energy(self, batch_ind=None, cubature=None, **kwargs):
        """
        The PL energy given in [1] is a poor approximation to the EP energy (although the gradients are ok, since
        the part they discard does not depends on the hyperparameters). Therefore, we can choose to use either
        the variational free energy or EP energy here.
        TODO: develop a PL energy approximation that reuses the linearisation quantities and matches GHS / UKS etc.
        [1] Garcia-Fernandez, Tronarp, Särkkä (2018) 'Gaussian process classification using posterior linearisation'
        """
        if batch_ind is None:
            batch_ind = np.arange(self.num_data)
            scale = 1
        else:
            scale = self.num_data / batch_ind.shape[0]

        # compute the cavity distribution
        cavity_mean, cavity_cov = self.cavity_distribution(None, 1.)  # TODO: check batch_ind is not required
        cav_mean_f, cav_cov_f = self.conditional_posterior_to_data(None, cavity_mean, cavity_cov)

        # calculate log marginal likelihood and the new sites via moment matching:
        # EP expected density is log E_q[p(y|f)]
        lZ, _, _ = vmap(self.likelihood.moment_match, (0, 0, 0, None, None))(
            self.Y[batch_ind],
            cav_mean_f[batch_ind],
            cav_cov_f[batch_ind],
            1.,
            cubature
        )

        if self.mask_y is not None:
            if np.squeeze(self.mask_y[batch_ind]).ndim != np.squeeze(lZ).ndim:
                raise NotImplementedError('masking in spatio-temporal models not implemented for EP')
            lZ = np.where(np.squeeze(self.mask_y[batch_ind]), 0., np.squeeze(lZ))  # apply mask

        pseudo_y, pseudo_var = self.compute_full_pseudo_lik()
        lZ_pseudo = mvn_logpdf_(
            pseudo_y,
            cavity_mean,
            pseudo_var + cavity_cov,
            self.mask_pseudo_y
        )

        if isinstance(self, SparseGP):
            pseudo_y, pseudo_var = self.compute_global_pseudo_lik()
            # TODO: check derivation for SparseGP. May take different form
            # lZ_post = np.sum(vmap(self.compute_log_lik)(pseudo_y, pseudo_var))
        # else:
        lZ_post = self.compute_log_lik(pseudo_y, pseudo_var)

        ep_energy = -(
                lZ_post
                + (scale * np.nansum(lZ) - np.nansum(lZ_pseudo))
        )

        return ep_energy


class Taylor(PosteriorLinearisation):  # TODO: inherits energy from PL - implement custom method that avoids cubature
    """
    Inference using analytical linearisation, i.e. a first order Taylor expansion. This is equivalent to
    the Extended Kalman Smoother when using a Markov GP.
    """

    def update_variational_params(self, batch_ind=None, lr=1., **kwargs):
        """
        """
        if batch_ind is None:
            batch_ind = np.arange(self.num_data)
        Y = self.Y[batch_ind]

        mean_f, cov_f = self.conditional_posterior_to_data(batch_ind)

        # calculate the Jacobian of the observation model w.r.t. function fₙ and noise term rₙ
        Jf, _, Jsigma, _ = vmap(self.likelihood.analytical_linearisation)(mean_f, np.zeros_like(Y))  # evaluate at mean

        obs_cov = np.eye(Y.shape[1])  # observation noise scale is w.l.o.g. 1
        sigma = Jsigma @ obs_cov @ transpose(Jsigma)
        likelihood_expectation, _ = vmap(self.likelihood.conditional_moments)(mean_f)
        residual = Y.reshape(likelihood_expectation.shape) - likelihood_expectation  # residual, yₙ-E[yₙ|fₙ]

        mask = np.isnan(residual)
        residual = np.where(mask, 0., residual)

        Jf_invsigma = transpose(solve(sigma, Jf))  # Jf^T inv(sigma)
        jacobian = Jf_invsigma @ residual
        hessian = -Jf_invsigma @ Jf

        # deal with missing data
        jacobian = np.where(mask, np.nan, jacobian)
        hessian = np.where(mask * np.eye(mask.shape[1]), np.nan, hessian)

        jacobian, hessian = self.conditional_data_to_posterior(jacobian, hessian)

        if mean_f.shape[1] == jacobian.shape[1]:
            return mean_f, jacobian, hessian
        else:
            ind = self.ind[batch_ind]
            return self.posterior_mean.value[ind], jacobian, hessian  # sparse Markov case


class ExtendedKalmanSmoother(Taylor):
    pass
