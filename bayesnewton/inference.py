import objax
import jax.numpy as np
from jax import vmap, Array
from .utils import (
    diag,
    transpose,
    inv_vmap,
    vmap_diag,
    solve,
    ensure_diagonal_positive_precision,
    mvn_logpdf_
)
from .likelihoods import Likelihood, MultiLatentLikelihood
from .basemodels import GaussianDistribution, SparseGP
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
    Y: Array
    ind: Array
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

        diff1 = np.mean(np.abs(nat1 - self.pseudo_likelihood.nat1))
        diff2 = np.mean(np.abs(nat2 - self.pseudo_likelihood.nat2))

        # ---- update the model variational parameters ----
        self.pseudo_likelihood.update_nat_params(
            nat1=(1 - lr) * self.pseudo_likelihood.nat1 + lr * nat1,
            nat2=(1 - lr) * self.pseudo_likelihood.nat2 + lr * nat2
        )

        self.update_posterior()  # recompute posterior with new params

        return (mean, jacobian, hessian), (diff1, diff2)  # output state to be used in linesearch methods

    def update_variational_params(self, batch_ind=None, lr=1., **kwargs):
        raise NotImplementedError

    def energy(self, batch_ind=None, **kwargs):
        raise NotImplementedError


class Newton(InferenceMixin):
    """
    Newton = Laplace
    """
    compute_kl: classmethod

    def update_variational_params(self, batch_ind=None, lr=1., ensure_psd=True, **kwargs):
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

        if ensure_psd:
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


Laplace = Newton


class VariationalInference(InferenceMixin):
    """
    Natural gradient VI (using the conjugate-computation VI approach)
    Refs:
        Khan & Lin 2017 "Conugate-computation variational inference - converting inference
                         in non-conjugate models in to inference in conjugate models"
        Chang, Wilkinson, Khan & Solin 2020 "Fast variational learning in state space Gaussian process models"
    """
    compute_kl: classmethod

    def update_variational_params(self, batch_ind=None, lr=1., cubature=None, ensure_psd=True, **kwargs):
        """
        """
        if batch_ind is None:
            batch_ind = np.arange(self.num_data)

        mean_f, cov_f = self.conditional_posterior_to_data(batch_ind)

        # VI expected density is expected log-likelihood: E_q[log p(y|f)]
        ell, dell_dm, d2ell_dm2 = vmap(self.likelihood.variational_expectation, (0, 0, 0, None))(
            self.Y[batch_ind],
            mean_f,
            cov_f,
            cubature
        )

        if ensure_psd:
            d2ell_dm2 = -ensure_diagonal_positive_precision(-d2ell_dm2)  # manual fix to avoid non-PSD precision

        jacobian, hessian = self.conditional_data_to_posterior(dell_dm, d2ell_dm2)

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

        # VI expected density is expected log-likelihood: E_q[log p(y|f)]
        ell, _, _ = vmap(self.likelihood.variational_expectation, (0, 0, 0, None))(
            self.Y[batch_ind],
            mean_f,
            cov_f,
            cubature
        )

        KL = self.compute_kl()  # KL[q(f)|p(f)]
        variational_free_energy = -(  # the variational free energy, i.e., the negative ELBO
            scale * np.nansum(ell)  # nansum accounts for missing data
            - KL
        )

        return variational_free_energy


class ExpectationPropagation(InferenceMixin):
    """
    Expectation propagation (EP)
    """
    power: float
    cavity_distribution: classmethod
    cavity_distribution_tied: classmethod
    compute_full_pseudo_lik: classmethod
    compute_log_lik: classmethod
    compute_ep_energy_terms: classmethod
    mask_y: Array
    mask_pseudo_y: Array

    def update_variational_params(self, batch_ind=None, lr=1., cubature=None, ensure_psd=True, **kwargs):
        """
        """
        if batch_ind is None:
            batch_ind = np.arange(self.num_data)

        # compute the cavity distribution
        if isinstance(self, SparseGP):
            # we use the tied version for sparse EP because it is much more efficient
            cavity_mean, cavity_cov = self.cavity_distribution_tied(batch_ind, self.power)
        else:
            cavity_mean, cavity_cov = self.cavity_distribution(batch_ind, self.power)

        cav_mean_f, cav_cov_f = self.conditional_posterior_to_data(batch_ind, cavity_mean, cavity_cov)

        # calculate log marginal likelihood and the new sites via moment matching:
        # EP expected density is the log expected likelihood: log E_q[p(y|f)]
        lel, dlel, d2lel = vmap(self.likelihood.moment_match, (0, 0, 0, None, None))(
            self.Y[batch_ind],
            cav_mean_f,
            cav_cov_f,
            self.power,
            cubature
        )

        cav_prec = inv_vmap(cav_cov_f)
        scale_factor = cav_prec @ inv_vmap(d2lel + cav_prec) / self.power  # this form guarantees symmetry

        dlel = scale_factor @ dlel
        d2lel = scale_factor @ d2lel
        if self.mask_pseudo_y is not None:
            # apply mask
            mask = self.mask_pseudo_y[batch_ind][..., None]
            dlel = np.where(mask, np.nan, dlel)
            d2lZ_masked = np.where(mask + transpose(mask), 0., d2lel)  # ensure masked entries are independent
            d2lel = np.where(diag(mask)[..., None], np.nan, d2lZ_masked)  # ensure masked entries return log like of 0

        if ensure_psd:
            d2lel = -ensure_diagonal_positive_precision(-d2lel)  # manual fix to avoid non-PSD precision

        jacobian, hessian = self.conditional_data_to_posterior(dlel, d2lel)

        if cav_mean_f.shape[1] == jacobian.shape[1]:
            return cav_mean_f, jacobian, hessian
        else:
            ind = self.ind[batch_ind]
            return cavity_mean[ind], jacobian, hessian  # sparse Markov case

    def energy(self, batch_ind=None, cubature=None, **kwargs):
        """
        """
        if batch_ind is None:
            batch_ind = np.arange(self.num_data)
            scale = 1.
        else:
            scale = self.num_data / batch_ind.shape[0]

        if isinstance(self, SparseGP):
            (cavity_mean, cavity_cov), lel_pseudo, lZ = self.compute_ep_energy_terms(batch_ind, self.power)
            cav_mean_f, cav_cov_f = self.conditional_posterior_to_data(batch_ind, cavity_mean, cavity_cov)
        else:
            # TODO: the batch indices could be handled better here
            (cavity_mean, cavity_cov), lel_pseudo, lZ = self.compute_ep_energy_terms(None, self.power)
            cav_mean_f, cav_cov_f = self.conditional_posterior_to_data(None, cavity_mean, cavity_cov)
            cav_mean_f, cav_cov_f = cav_mean_f[batch_ind], cav_cov_f[batch_ind]

        # EP expected density is log expected likelihood: log E_q[p(y|f)]
        lel, _, _ = vmap(self.likelihood.moment_match, (0, 0, 0, None, None))(
            self.Y[batch_ind],
            cav_mean_f,
            cav_cov_f,
            self.power,
            cubature
        )

        if self.mask_y is not None:
            # TODO: this assumes masking is implemented in MultiLatentLikelihood - generalise
            if not isinstance(self.likelihood, MultiLatentLikelihood):
                if np.squeeze(self.mask_y[batch_ind]).ndim != np.squeeze(lel).ndim:
                    raise NotImplementedError('masking in spatio-temporal models not implemented for EP')
                lel = np.where(np.squeeze(self.mask_y[batch_ind]), 0., np.squeeze(lel))  # apply mask

        ep_energy = -(
            lZ
            + 1. / self.power * (scale * np.nansum(lel) - np.nansum(lel_pseudo))
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
    mask_y: Array
    mask_pseudo_y: Array

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

        # deal with missing data
        mask = np.isnan(residual)
        residual = np.where(mask, 0., residual)
        omega = np.where(mask + transpose(mask), 0., omega)
        omega = np.where(vmap_diag(mask[..., 0]), 1e6, omega)

        dmu_omega = transpose(solve(omega, d_mu))  # d_mu^T inv(omega)
        jacobian = dmu_omega @ residual
        hessian = -dmu_omega @ d_mu

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
            # TODO: this assumes masking is implemented in MultiLatentLikelihood - generalise
            if not isinstance(self.likelihood, MultiLatentLikelihood):
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


class PosteriorLinearisation2ndOrder(PosteriorLinearisation):
    """
    """
    # TODO: remove these when possible
    compute_full_pseudo_lik: classmethod
    mask_y: Array
    mask_pseudo_y: Array

    def update_variational_params(self, batch_ind=None, lr=1., cubature=None, ensure_psd=True, **kwargs):
        """
        """
        if batch_ind is None:
            batch_ind = np.arange(self.num_data)

        mean_f, cov_f = self.conditional_posterior_to_data(batch_ind)

        log_target, jacobian, hessian = vmap(self.likelihood.statistical_linear_regression_newton, (0, 0, 0, None))(
            self.Y[batch_ind],
            mean_f,
            cov_f,
            cubature,
        )

        if ensure_psd:
            hessian = -ensure_diagonal_positive_precision(-hessian)  # manual fix to avoid non-PSD precision

        jacobian, hessian = self.conditional_data_to_posterior(jacobian, hessian)

        if mean_f.shape[1] == jacobian.shape[1]:
            return mean_f, jacobian, hessian
        else:
            ind = self.ind[batch_ind]
            return self.posterior_mean.value[ind], jacobian, hessian  # sparse Markov case


class Taylor(Newton):
    """
    Inference using analytical linearisation, i.e. a first order Taylor expansion. This is equivalent to
    the Extended Kalman Smoother when using a Markov GP.

    Note: this class inherits the energy from the Laplace method. This is suitable because Taylor = a Gauss-Newton
    approximation to Laplace (which uses Newton's method). Their energy should be equivalent since the only
    difference between the algorithms is the way the Hessian of the likelihood is computed / approximated, which
    does not effect the energy.
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


ExtendedKalmanSmoother = Taylor


class GaussNewton(Newton):
    """
    Gauss-Newton
    """

    def update_variational_params(self, batch_ind=None, lr=1., cubature=None, **kwargs):
        """
        """
        if batch_ind is None:
            batch_ind = np.arange(self.num_data)

        mean_f, cov_f = self.conditional_posterior_to_data(batch_ind)

        log_target, jacobian, hessian = vmap(self.likelihood.gauss_newton, (0, 0))(
            self.Y[batch_ind],
            mean_f
        )

        jacobian, hessian = self.conditional_data_to_posterior(jacobian, hessian)

        if mean_f.shape[1] == jacobian.shape[1]:
            return mean_f, jacobian, hessian
        else:
            ind = self.ind[batch_ind]
            return self.posterior_mean.value[ind], jacobian, hessian  # sparse Markov case


class VariationalGaussNewton(VariationalInference):
    """
    Variational Gauss-Newton
    """

    def update_variational_params(self, batch_ind=None, lr=1., cubature=None, **kwargs):
        """
        """
        if batch_ind is None:
            batch_ind = np.arange(self.num_data)

        mean_f, cov_f = self.conditional_posterior_to_data(batch_ind)

        log_target, jacobian, hessian = vmap(self.likelihood.variational_gauss_newton, (0, 0, 0, None))(
            self.Y[batch_ind],
            mean_f,
            cov_f,
            cubature
        )

        jacobian, hessian = self.conditional_data_to_posterior(jacobian, hessian)

        if mean_f.shape[1] == jacobian.shape[1]:
            return mean_f, jacobian, hessian
        else:
            ind = self.ind[batch_ind]
            return self.posterior_mean.value[ind], jacobian, hessian  # sparse Markov case


class PosteriorLinearisation2ndOrderGaussNewton(PosteriorLinearisation):
    """
    """
    # TODO: remove these when possible
    compute_full_pseudo_lik: classmethod
    mask_y: Array
    mask_pseudo_y: Array

    def update_variational_params(self, batch_ind=None, lr=1., cubature=None, **kwargs):
        """
        """
        if batch_ind is None:
            batch_ind = np.arange(self.num_data)

        mean_f, cov_f = self.conditional_posterior_to_data(batch_ind)

        log_target, jacobian, hessian = vmap(self.likelihood.statistical_linear_regression_gauss_newton, (0, 0, 0, None))(
            self.Y[batch_ind],
            mean_f,
            cov_f,
            cubature,
        )

        jacobian, hessian = self.conditional_data_to_posterior(jacobian, hessian)

        if mean_f.shape[1] == jacobian.shape[1]:
            return mean_f, jacobian, hessian
        else:
            ind = self.ind[batch_ind]
            return self.posterior_mean.value[ind], jacobian, hessian  # sparse Markov case


class NewtonRiemann(Newton):
    """
    """

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

        jacobian, hessian = self.conditional_data_to_posterior(jacobian[..., None], hessian)

        if mean_f.shape[1] == jacobian.shape[1]:
            G = self.pseudo_likelihood.nat2 + hessian  # TODO: sort out batching here?
            hessian_psd = hessian - 0.5 * lr * (G @ self.pseudo_likelihood.covariance @ G)
            return mean_f, jacobian, hessian_psd
        else:
            ind = self.ind[batch_ind]
            G = self.pseudo_likelihood.nat2[ind] + hessian  # TODO: sort out batching here?
            hessian_psd = hessian - 0.5 * lr * (G @ self.pseudo_likelihood.covariance[ind] @ G)
            return self.posterior_mean.value[ind], jacobian, hessian_psd  # sparse Markov case


LaplaceRiemann = NewtonRiemann


class VariationalInferenceRiemann(VariationalInference):
    """
    Natural gradient VI (using the conjugate-computation VI approach) with PSD constraints via Riemannian gradients
    Refs:
        Lin, Schmidt & Khan 2020 "Handling the Positive-Definite Constraint in the Bayesian Learning Rule"
        Khan & Lin 2017 "Conugate-computation variational inference - converting inference
                         in non-conjugate models in to inference in conjugate models"
        Chang, Wilkinson, Khan & Solin 2020 "Fast variational learning in state space Gaussian process models"
    TODO: implement grouped update to enable sparse markov inference
    """

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

        jacobian, hessian = self.conditional_data_to_posterior(dE_dm, d2E_dm2)

        if mean_f.shape[1] == jacobian.shape[1]:
            G = self.pseudo_likelihood.nat2 + hessian  # TODO: sort out batching here?
            hessian_psd = hessian - 0.5 * lr * (G @ self.pseudo_likelihood.covariance @ G)
            return mean_f, jacobian, hessian_psd
        else:
            ind = self.ind[batch_ind]
            G = self.pseudo_likelihood.nat2[ind] + hessian  # TODO: sort out batching here?
            hessian_psd = hessian - 0.5 * lr * (G @ self.pseudo_likelihood.covariance[ind] @ G)
            return self.posterior_mean.value[ind], jacobian, hessian_psd  # sparse Markov case


class ExpectationPropagationRiemann(ExpectationPropagation):
    """
    Expectation propagation (EP) with PSD constraints via Riemannian gradients
    """

    def update_variational_params(self, batch_ind=None, lr=1., cubature=None, **kwargs):
        """
        TODO: will not currently work with SparseGP because cavity_cov is a vector (SparseGP and SparseMarkovGP use different parameterisations)
        """
        if batch_ind is None:
            batch_ind = np.arange(self.num_data)

        # compute the cavity distribution
        cavity_mean, cavity_cov = self.cavity_distribution(batch_ind, self.power)
        cav_mean_f, cav_cov_f = self.conditional_posterior_to_data(batch_ind, cavity_mean, cavity_cov)

        # calculate log marginal likelihood and the new sites via moment matching:
        # EP expected density is log E_q[p(y|f)]
        lZ, dlZ, d2lZ = vmap(self.likelihood.moment_match, (0, 0, 0, None, None))(
            self.Y[batch_ind],
            cav_mean_f,
            cav_cov_f,
            self.power,
            cubature
        )

        cav_prec = inv_vmap(cav_cov_f)
        scale_factor = cav_prec @ inv_vmap(d2lZ + cav_prec) / self.power  # this form guarantees symmetry

        dlZ = scale_factor @ dlZ
        d2lZ = scale_factor @ d2lZ
        if self.mask_pseudo_y is not None:
            # apply mask
            mask = self.mask_pseudo_y[batch_ind][..., None]
            dlZ = np.where(mask, np.nan, dlZ)
            d2lZ_masked = np.where(mask + transpose(mask), 0., d2lZ)  # ensure masked entries are independent
            d2lZ = np.where(diag(mask)[..., None], np.nan, d2lZ_masked)  # ensure masked entries return nan

        jacobian, hessian = self.conditional_data_to_posterior(dlZ, d2lZ)

        if cav_mean_f.shape[1] == jacobian.shape[1]:
            G = self.pseudo_likelihood.nat2 + hessian  # TODO: sort out batching here?
            hessian_psd = hessian - 0.5 * lr * (G @ self.pseudo_likelihood.covariance @ G)
            return cav_mean_f, jacobian, hessian_psd
        else:
            ind = self.ind[batch_ind]
            G = self.pseudo_likelihood.nat2[ind] + hessian  # TODO: sort out batching here?
            hessian_psd = hessian - 0.5 * lr * (G @ self.pseudo_likelihood.covariance[ind] @ G)
            return cavity_mean[ind], jacobian, hessian_psd  # sparse Markov case


class PosteriorLinearisation2ndOrderRiemann(PosteriorLinearisation2ndOrder):
    """
    """

    def update_variational_params(self, batch_ind=None, lr=1., cubature=None, **kwargs):
        """
        """
        if batch_ind is None:
            batch_ind = np.arange(self.num_data)

        mean_f, cov_f = self.conditional_posterior_to_data(batch_ind)

        log_target, jacobian, hessian = vmap(self.likelihood.statistical_linear_regression_newton, (0, 0, 0, None))(
            self.Y[batch_ind],
            mean_f,
            cov_f,
            cubature,
        )

        jacobian, hessian = self.conditional_data_to_posterior(jacobian, hessian)

        if mean_f.shape[1] == jacobian.shape[1]:
            G = self.pseudo_likelihood.nat2 + hessian  # TODO: sort out batching here?
            hessian_psd = hessian - 0.5 * lr * (G @ self.pseudo_likelihood.covariance @ G)
            return mean_f, jacobian, hessian_psd
        else:
            ind = self.ind[batch_ind]
            G = self.pseudo_likelihood.nat2[ind] + hessian  # TODO: sort out batching here?
            hessian_psd = hessian - 0.5 * lr * (G @ self.pseudo_likelihood.covariance[ind] @ G)
            return self.posterior_mean.value[ind], jacobian, hessian_psd  # sparse Markov case


def bfgs(mean, jacobian, mean_prev, jacobian_prev, B_prev):
    """ The BFGS update is guaranteed to result in PSD updates only if sg < 0 """
    g = jacobian - jacobian_prev
    s = mean - mean_prev
    sBs = transpose(s) @ B_prev @ s
    sg = transpose(s) @ g
    B_new = (
        B_prev
        - (B_prev @ s @ transpose(s) @ B_prev) / sBs
        + (g @ transpose(g)) / sg
    )
    B_new = np.where(sg < -1e-14, B_new, B_prev)  # don't update the hessian if non-psd
    return B_new


def damped_bfgs(mean, jacobian, mean_prev, jacobian_prev, B_prev, damping=0.8):
    """ The damped BFGS update is guaranteed to result in PSD updates """
    g = jacobian - jacobian_prev
    s = mean - mean_prev
    sBs = transpose(s) @ B_prev @ s
    sg = transpose(s) @ g
    theta = np.where(sg > (1 - damping) * sBs,
                     damping * sBs / (sBs - sg),
                     np.ones_like(sBs))
    r = theta * g + (1 - theta) * B_prev @ s
    sr = transpose(s) @ r
    B_new = (
        B_prev
        - (B_prev @ s @ transpose(s) @ B_prev) / sBs
        + (r @ transpose(r)) / sr
    )
    # sr guaranteed to be positive except for at initial point, or for points where theta=1, so we still check:
    B_new = np.where(sr < -1e-14, B_new, B_prev)
    return B_new  # convert back to NSD


def damped_bfgs_modified(mean, jacobian, mean_prev, jacobian_prev, B_prev, damping=0.8):
    """
    The damped BFGS update is guaranteed to result in PSD updates
    This version uses the undamped BFGS if damping = 1
    """
    g = jacobian - jacobian_prev
    s = mean - mean_prev
    sBs = transpose(s) @ B_prev @ s
    sg = transpose(s) @ g
    theta = np.where(sg > (1 - damping) * sBs,
                     damping * sBs / (sBs - sg),
                     np.ones_like(sBs))
    theta = np.where(damping > 0.99, np.ones_like(theta), theta)
    r = theta * g + (1 - theta) * B_prev @ s
    sr = transpose(s) @ r
    B_new = (
        B_prev
        - (B_prev @ s @ transpose(s) @ B_prev) / sBs
        + (r @ transpose(r)) / sr
    )
    # sr guaranteed to be positive except for at initial point, or for points where theta=1, so we still check:
    B_new = np.where(sr < -1e-14, B_new, B_prev)
    return B_new  # convert back to NSD


class QuasiNewtonBase(abc.ABC):
    num_data: float
    func_dim: float
    fullcov: bool
    update_posterior: classmethod
    group_natural_params: classmethod
    update_variational_params: classmethod
    pseudo_likelihood: GaussianDistribution
    mean_prev: objax.StateVar
    jacobian_prev: objax.StateVar
    hessian_approx: objax.StateVar
    posterior_variance: objax.StateVar

    def inference(self, lr=1., batch_ind=None, **kwargs):

        if (batch_ind is None) or (batch_ind.shape[0] == self.num_data):
            batch_ind = None

        self.update_posterior()  # make sure the posterior is up to date

        # use the chosen inference method (VI, EP, ...) to compute the necessary terms for the parameter update
        mean, jacobian, hessian, quasi_newton_state = self.update_variational_params(batch_ind, lr, **kwargs)

        # update tracked quasi-newton params
        quasi_newton_mean, quasi_newton_jacobian, quasi_newton_hessian = quasi_newton_state
        self.mean_prev.value = quasi_newton_mean
        self.jacobian_prev.value = quasi_newton_jacobian
        self.hessian_approx.value = (1 - lr) * self.hessian_approx.value + lr * quasi_newton_hessian

        # ---- Newton update ----
        nat1_n, nat2_n = newton_update(mean, jacobian, hessian)
        # -----------------------
        nat1, nat2 = self.group_natural_params(nat1_n, nat2_n, batch_ind)  # only required for SparseMarkov models

        diff1 = np.mean(np.abs(nat1 - self.pseudo_likelihood.nat1))
        diff2 = np.mean(np.abs(nat2 - self.pseudo_likelihood.nat2))

        # ---- update the model variational parameters ----
        # for quasi-Newton, the damping is applied during the BFGS update
        self.pseudo_likelihood.update_nat_params(
            nat1=(1 - lr) * self.pseudo_likelihood.nat1 + lr * nat1,
            # nat1=nat1,
            nat2=(1 - lr) * self.pseudo_likelihood.nat2 + lr * nat2
            # nat2=nat2
        )

        self.update_posterior()  # recompute posterior with new params

        # return (mean, jacobian, hessian), quasi_newton_state  # output state to be used in linesearch methods
        return (mean, jacobian, hessian), (diff1, diff2)  # output state to be used in linesearch methods


class QuasiNewton(QuasiNewtonBase, Newton):
    """
    TODO: implement grouped update to enable sparse markov inference
    """

    def update_variational_params(self, batch_ind=None, lr=1., damping=1., **kwargs):
        """
        """
        if batch_ind is None:
            batch_ind = np.arange(self.num_data)
        ind = self.ind[batch_ind]

        mean_f, _ = self.conditional_posterior_to_data(batch_ind)

        # Laplace approximates the expected density with a point estimate at the posterior mean: log p(y|f=m)
        log_lik, jacobian, _ = vmap(self.likelihood.log_likelihood_gradients)(  # parallel
            self.Y[batch_ind],
            mean_f
        )

        jacobian, _ = self.conditional_data_to_posterior(jacobian[..., None], _)

        B = damped_bfgs_modified(
            mean_f, jacobian,
            self.mean_prev.value[ind], self.jacobian_prev.value[ind],
            self.hessian_approx.value[ind],
            damping=damping
        )

        if self.mean_prev.value.shape[0] != mean_f.shape[0]:
            B = self.hessian_approx.value.at[ind].set(B)
            jacobian = self.jacobian_prev.value.at[ind].set(jacobian) 
            mean_f = self.mean_prev.value.at[ind].set(mean_f)

        return mean_f, jacobian, B, (mean_f, jacobian, B)


class VariationalQuasiNewton(QuasiNewtonBase, VariationalInference):
    """
    TODO: implement grouped update to enable sparse markov inference
    """

    def update_variational_params(self, batch_ind=None, lr=1., damping=1., cubature=None, **kwargs):
        """
        """
        if batch_ind is None:
            batch_ind = np.arange(self.num_data)
        ind = self.ind[batch_ind]

        mean_f, cov_f = self.conditional_posterior_to_data(batch_ind)

        # VI expected density is E_q[log p(y|f)]
        expected_density, dE_dm, d2E_dm2 = vmap(self.likelihood.variational_expectation, (0, 0, 0, None))(
            self.Y[batch_ind],
            mean_f,
            cov_f,
            cubature
        )

        dE_dv = 0.5 * d2E_dm2

        jacobian, _ = self.conditional_data_to_posterior(dE_dm, d2E_dm2)

        if self.fullcov:
            mean_var = np.concatenate([mean_f, cov_f.reshape(self.num_data, -1, 1)], axis=1)
            jacobian_mean_var = np.concatenate([dE_dm, dE_dv.reshape(self.num_data, -1, 1)], axis=1)
        else:
            mean_var = np.concatenate([mean_f, diag(cov_f)[..., None]], axis=1)
            jacobian_mean_var = np.concatenate([dE_dm, diag(dE_dv)[..., None]], axis=1)

        B = damped_bfgs_modified(
            mean_var, jacobian_mean_var,
            self.mean_prev.value[ind], self.jacobian_prev.value[ind],
            self.hessian_approx.value[ind],
            damping=damping
        )

        if self.mean_prev.value.shape[0] != mean_f.shape[0]:
            B = self.hessian_approx.value.at[ind].set(B)
            jacobian = self.jacobian_prev.value.at[ind].set(jacobian)
            mean_f = self.mean_prev.value.at[ind].set(mean_f)

        return mean_f, jacobian, B[:, :self.func_dim, :self.func_dim], (mean_var, jacobian_mean_var, B)


class ExpectationPropagationQuasiNewton(QuasiNewtonBase, ExpectationPropagation):
    """
    TODO: implement grouped update to enable sparse markov inference
    """

    def update_variational_params(self, batch_ind=None, lr=1., damping=1., cubature=None, **kwargs):
        """
        """
        if batch_ind is None:
            batch_ind = np.arange(self.num_data)
        ind = self.ind[batch_ind]

        # compute the cavity distribution
        cavity_mean, cavity_cov = self.cavity_distribution(batch_ind, self.power)
        cav_mean_f, cav_cov_f = self.conditional_posterior_to_data(batch_ind, cavity_mean, cavity_cov)

        # calculate log marginal likelihood and the new sites via moment matching:
        # EP expected density is log E_q[p(y|f)]
        lZ, dlZ_dm, d2lZ_dm, dlZ_dv = vmap(self.likelihood.moment_match_dv, (0, 0, 0, None, None))(
            self.Y[batch_ind],
            cav_mean_f,
            cav_cov_f,
            self.power,
            cubature
        )

        jacobian_unscaled, _ = self.conditional_data_to_posterior(dlZ_dm, d2lZ_dm)
        if self.fullcov:
            cavity_mean_var = np.concatenate([cav_mean_f, cav_cov_f.reshape(self.num_data, -1, 1)], axis=1)
            jacobian_unscaled_mean_var = np.concatenate([dlZ_dm, dlZ_dv.reshape(self.num_data, -1, 1)], axis=1)
        else:
            cavity_mean_var = np.concatenate([cav_mean_f, diag(cav_cov_f)[..., None]], axis=1)
            jacobian_unscaled_mean_var = np.concatenate([dlZ_dm, diag(dlZ_dv)[..., None]], axis=1)

        B = damped_bfgs_modified(
            cavity_mean_var, jacobian_unscaled_mean_var,
            self.mean_prev.value[ind], self.jacobian_prev.value[ind],
            self.hessian_approx.value[ind],
            damping=damping
        )

        cav_prec = inv_vmap(cav_cov_f)
        scale_factor = cav_prec @ inv_vmap(-B[:, :self.func_dim, :self.func_dim] + cav_prec) / self.power

        jacobian = scale_factor @ jacobian_unscaled
        hessian = scale_factor @ B[:, :self.func_dim, :self.func_dim]

        if self.mean_prev.value.shape[0] != cav_mean_f.shape[0]:
            B = self.hessian_approx.value.at[ind].set(B)
            jacobian = self.jacobian_prev.value.at[ind].set(jacobian) 
            cav_mean_f = self.mean_prev.value.at[ind].set(cav_mean_f)


        return cav_mean_f, jacobian, hessian, (cavity_mean_var, jacobian_unscaled_mean_var, B)


# class PosteriorLinearisationQuasiNewton(QuasiNewtonBase, PosteriorLinearisation):
#     """
#     Quasi-Newton (BFGS) Posterior Linearisation (PL)
#     TODO: implement grouped update to enable sparse markov inference
#     """
#
#     def update_variational_params(self, batch_ind=None, lr=1., damping=1., cubature=None, **kwargs):
#         """
#         """
#         if batch_ind is None:
#             batch_ind = np.arange(self.num_data)
#         ind = self.ind[batch_ind]
#
#         mean_f, cov_f = self.conditional_posterior_to_data(batch_ind)
#
#         mu, omega, dmu_dm, d2mu_dm2 = vmap(self.likelihood.statistical_linear_regression, (0, 0, None))(
#             mean_f,
#             cov_f,
#             cubature
#         )
#         dmu_dv = 0.5 * d2mu_dm2
#         dmu_dv = dmu_dv.reshape(dmu_dv.shape[0], dmu_dv.shape[1], -1)
#
#         residual = self.Y[batch_ind].reshape(mu.shape) - mu
#
#         # deal with missing data
#         mask = np.isnan(residual)
#         residual = np.where(mask, 0., residual)
#         omega = np.where(mask + transpose(mask), 0., omega)
#         omega = np.where(vmap_diag(mask[..., 0]), 1e6, omega)
#
#         dmu_omega = transpose(solve(omega, dmu_dm))  # dmu_dm^T @ inv(omega)
#         jacobian = dmu_omega @ residual
#         hessian_not_used = -dmu_omega @ dmu_dm
#         jacobian_var = transpose(solve(omega, dmu_dv)) @ residual
#
#         jacobian, _ = self.conditional_data_to_posterior(jacobian, hessian_not_used)
#
#         if self.fullcov:
#             mean_var = np.concatenate([mean_f, cov_f.reshape(self.num_data, -1, 1)], axis=1)
#             jacobian_mean_var = np.concatenate([
#                 jacobian.reshape(self.num_data, -1, 1),
#                 jacobian_var.reshape(self.num_data, -1, 1)
#             ], axis=1)
#         else:
#             mean_var = np.concatenate([mean_f, diag(cov_f)[..., None]], axis=1)
#             d = mean_f.shape[1]
#             jacobian_mean_var = np.concatenate([
#                 jacobian,
#                 diag(jacobian_var.reshape(self.num_data, d, d))[..., None]
#             ], axis=1)
#
#         B = damped_bfgs_modified(
#             mean_var, jacobian_mean_var,
#             self.mean_prev.value[ind], self.jacobian_prev.value[ind],
#             self.hessian_approx.value[ind],
#             damping=damping
#         )
#
#         if self.mean_prev.value.shape[0] != mean_f.shape[0]:
#             B = index_update(self.hessian_approx.value, index[ind], B)
#             jacobian = index_update(self.jacobian_prev.value, index[ind], jacobian)
#             mean_f = index_update(self.mean_prev.value, index[ind], mean_f)
#
#         return mean_f, jacobian, B[:, :self.func_dim, :self.func_dim], (mean_var, jacobian_mean_var, B)


class PosteriorLinearisation2ndOrderQuasiNewton(QuasiNewtonBase, PosteriorLinearisation):
    """
    """

    def update_variational_params(self, batch_ind=None, lr=1., damping=1., cubature=None, **kwargs):
        """
        """
        if batch_ind is None:
            batch_ind = np.arange(self.num_data)
        ind = self.ind[batch_ind]

        mean_f, cov_f = self.conditional_posterior_to_data(batch_ind)

        log_target, dmu_dm, d2mu_dm2 = vmap(self.likelihood.statistical_linear_regression_quasi_newton, (0, 0, 0, None))(
            self.Y[batch_ind],
            mean_f,
            cov_f,
            cubature,
        )
        dmu_dv = 0.5 * d2mu_dm2

        jacobian, _ = self.conditional_data_to_posterior(dmu_dm, dmu_dv)

        if self.fullcov:
            mean_var = np.concatenate([mean_f, cov_f.reshape(self.num_data, -1, 1)], axis=1)
            jacobian_mean_var = np.concatenate([dmu_dm, dmu_dv.reshape(self.num_data, -1, 1)], axis=1)
        else:
            mean_var = np.concatenate([mean_f, diag(cov_f)[..., None]], axis=1)
            jacobian_mean_var = np.concatenate([dmu_dm, diag(dmu_dv)[..., None]], axis=1)

        B = damped_bfgs_modified(
            mean_var, jacobian_mean_var,
            self.mean_prev.value[ind], self.jacobian_prev.value[ind],
            self.hessian_approx.value[ind],
            damping=damping
        )

        if self.mean_prev.value.shape[0] != mean_f.shape[0]:
            B = self.hessian_approx.value.at[ind].set(B)
            jacobian = self.jacobian_prev.value.at[ind].set(jacobian) 
            mean_f = self.mean_prev.value.at[ind].set(mean_f)

        return mean_f, jacobian, B[:, :self.func_dim, :self.func_dim], (mean_var, jacobian_mean_var, B)
