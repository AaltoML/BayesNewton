import objax
import jax.numpy as np
from jax import vmap
from .cubature import GaussHermite
from .utils import (
    diag,
    transpose,
    inv,
    solve,
    ensure_positive_precision,
    ensure_diagonal_positive_precision,
    mvn_logpdf,
    pep_constant
)
import math

LOG2PI = math.log(2 * math.pi)


def newton_update(mean, jacobian, hessian):
    """
    Applies one step of Newton's method to update the pseudo_likelihood parameters
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


class Inference(objax.Module):
    """
    The approximate inference class.
    Each approximate inference scheme implements an 'update' method which is called during
    inference in order to update the local likelihood approximation (the sites).
    """
    def __init__(self,
                 cubature=GaussHermite()):
        self.cubature = cubature

    def __call__(self, model, lr=1., batch_ind=None):

        if (batch_ind is None) or (batch_ind.shape[0] == model.num_data):
            batch_ind = None

        model.update_posterior()  # make sure the posterior is up to date

        # use the chosen inference method (VI, EP, ...) to compute the necessary terms for the parameter update
        mean, jacobian, hessian = self.update(model, batch_ind, lr)
        # ---- Newton update ----
        nat1_n, nat2_n = newton_update(mean, jacobian, hessian)  # parallel operation
        # -----------------------
        nat1, nat2 = model.group_natural_params(nat1_n, nat2_n, batch_ind)  # sequential / batch operation

        # apply the parameter update and return the energy
        # ---- update the model parameters ----
        model.pseudo_likelihood_nat1.value = (
            (1 - lr) * model.pseudo_likelihood_nat1.value
            + lr * nat1
        )
        model.pseudo_likelihood_nat2.value = (
            (1 - lr) * model.pseudo_likelihood_nat2.value
            + lr * nat2
        )
        model.set_pseudo_likelihood()  # update mean and covariance

        model.update_posterior()  # recompute posterior with new params

    def update(self, model, batch_ind=None, lr=1.):
        raise NotImplementedError

    def energy(self, model, batch_ind=None):
        return model.filter_energy()


class Laplace(Inference):
    """
    """
    def __init__(self):
        super().__init__()
        self.name = 'Laplace / Newton\'s Algorithm (NA)'

    def update(self, model, batch_ind=None, lr=1.):
        """
        """
        if batch_ind is None:
            batch_ind = np.arange(model.num_data)

        mean_f, _ = model.conditional_posterior_to_data(batch_ind)

        # Laplace approximates the expected density with a point estimate at the posterior mean: log p(y|f=m)
        log_lik, jacobian, hessian = vmap(model.likelihood.log_likelihood_gradients)(  # parallel
            model.Y[batch_ind],
            mean_f
        )

        hessian = -ensure_positive_precision(-hessian)  # manual fix to avoid non-PSD precision

        jacobian, hessian = model.conditional_data_to_posterior(jacobian[..., None], hessian)

        if mean_f.shape[1] == jacobian.shape[1]:
            return mean_f, jacobian, hessian
        else:
            ind = model.ind[batch_ind]
            return model.posterior_mean.value[ind], jacobian, hessian  # sparse Markov case

    def energy(self, model, batch_ind=None):
        """
        """
        if batch_ind is None:
            batch_ind = np.arange(model.num_data)
            scale = 1
        else:
            scale = model.num_data / batch_ind.shape[0]

        mean_f, _ = model.conditional_posterior_to_data(batch_ind)

        # Laplace approximates the expected density with a point estimate at the posterior mean: log p(y|f=m)
        log_lik, _, _ = vmap(model.likelihood.log_likelihood_gradients)(  # parallel
            model.Y[batch_ind],
            mean_f
        )

        KL = model.compute_kl()  # KL[q(f)|p(f)]
        laplace_energy = -(  # Laplace approximation to the negative log marginal likelihood
            scale * np.nansum(log_lik)  # nansum accounts for missing data
            - KL
        )

        return laplace_energy


class VariationalInference(Inference):
    """
    Natural gradient VI (using the conjugate-computation VI approach)
    Refs:
        Khan & Lin 2017 "Conugate-computation variational inference - converting inference
                         in non-conjugate models in to inference in conjugate models"
        Chang, Wilkinson, Khan & Solin 2020 "Fast variational learning in state space Gaussian process models"
    """
    def __init__(self,
                 cubature=GaussHermite()):
        super().__init__(cubature=cubature)
        self.name = 'Variational Inference (VI)'

    def update(self, model, batch_ind=None, lr=1.):
        """
        """
        if batch_ind is None:
            batch_ind = np.arange(model.num_data)

        mean_f, cov_f = model.conditional_posterior_to_data(batch_ind)

        # VI expected density is E_q[log p(y|f)]
        expected_density, dE_dm, d2E_dm2 = vmap(model.likelihood.variational_expectation, (0, 0, 0, None))(
            model.Y[batch_ind],
            mean_f,
            cov_f,
            self.cubature
        )

        d2E_dm2 = -ensure_diagonal_positive_precision(-d2E_dm2)  # manual fix to avoid non-PSD precision

        jacobian, hessian = model.conditional_data_to_posterior(dE_dm, d2E_dm2)

        if mean_f.shape[1] == jacobian.shape[1]:
            return mean_f, jacobian, hessian
        else:
            ind = model.ind[batch_ind]
            return model.posterior_mean.value[ind], jacobian, hessian  # sparse Markov case

    def energy(self, model, batch_ind=None):
        """
        """
        if batch_ind is None:
            batch_ind = np.arange(model.num_data)
            scale = 1
        else:
            scale = model.num_data / batch_ind.shape[0]

        mean_f, cov_f = model.conditional_posterior_to_data(batch_ind)

        # VI expected density is E_q[log p(y|f)]
        expected_density, _, _ = vmap(model.likelihood.variational_expectation, (0, 0, 0, None))(
            model.Y[batch_ind],
            mean_f,
            cov_f,
            self.cubature
        )

        KL = model.compute_kl()  # KL[q(f)|p(f)]
        variational_free_energy = -(  # the variational free energy, i.e., the negative ELBO
            scale * np.nansum(expected_density)  # nansum accounts for missing data
            - KL
        )

        return variational_free_energy


class ExpectationPropagation(Inference):
    """
    Expectation propagation (EP)
    """
    def __init__(self,
                 power=1.0,
                 cubature=GaussHermite()):
        self.power = power
        super().__init__(cubature=cubature)
        self.name = 'Expectation Propagation (EP)'

    def update(self, model, batch_ind=None, lr=1.):
        """
        TODO: will not currently work with SparseGP because cavity_cov is a vector (SparseGP and SparseMarkovGP use different parameterisations)
        """
        if batch_ind is None:
            batch_ind = np.arange(model.num_data)

        # compute the cavity distribution
        cavity_mean, cavity_cov = model.cavity_distribution(batch_ind, self.power)
        cav_mean_f, cav_cov_f = model.conditional_posterior_to_data(batch_ind, cavity_mean, cavity_cov)

        # calculate log marginal likelihood and the new sites via moment matching:
        # EP expected density is log E_q[p(y|f)]
        lZ, dlZ, d2lZ = vmap(model.likelihood.moment_match, (0, 0, 0, None, None))(
            model.Y[batch_ind],
            cav_mean_f,
            cav_cov_f,
            self.power,
            self.cubature
        )

        cav_prec = vmap(inv)(cav_cov_f)
        scale_factor = cav_prec @ vmap(inv)(d2lZ + cav_prec) / self.power
        dlZ = scale_factor @ dlZ
        d2lZ = scale_factor @ d2lZ
        if model.mask is not None:
            # apply mask
            mask = model.mask[batch_ind][..., None]
            dlZ = np.where(mask, np.nan, dlZ)
            d2lZ_masked = np.where(mask + transpose(mask), 0., d2lZ)  # ensure masked entries are independent
            d2lZ = np.where(diag(mask)[..., None], np.nan, d2lZ_masked)  # ensure masked entries return log like of 0

        d2lZ = -ensure_diagonal_positive_precision(-d2lZ)  # manual fix to avoid non-PSD precision

        jacobian, hessian = model.conditional_data_to_posterior(dlZ, d2lZ)

        if cav_mean_f.shape[1] == jacobian.shape[1]:
            return cav_mean_f, jacobian, hessian
        else:
            ind = model.ind[batch_ind]
            return cavity_mean[ind], jacobian, hessian  # sparse Markov case

    def energy(self, model, batch_ind=None):
        """
        TODO: implement for SparseGP
        """
        if batch_ind is None:
            batch_ind = np.arange(model.num_data)
            scale = 1
        else:
            scale = model.num_data / batch_ind.shape[0]

        # compute the cavity distribution
        cavity_mean, cavity_cov = model.cavity_distribution(None, self.power)
        cav_mean_f, cav_cov_f = model.conditional_posterior_to_data(None, cavity_mean, cavity_cov)

        # calculate log marginal likelihood and the new sites via moment matching:
        # EP expected density is log E_q[p(y|f)]
        lZ, _, _ = vmap(model.likelihood.moment_match, (0, 0, 0, None, None))(
            model.Y[batch_ind],
            cav_mean_f[batch_ind],
            cav_cov_f[batch_ind],
            self.power,
            self.cubature
        )

        mask = model.mask  # [batch_ind]
        if model.mask is not None:
            if np.squeeze(mask[batch_ind]).ndim != np.squeeze(lZ).ndim:
                raise NotImplementedError('masking in spatio-temporal models not implemented for EP')
            lZ = np.where(np.squeeze(mask[batch_ind]), 0., np.squeeze(lZ))  # apply mask
            if mask.shape[1] != cavity_cov.shape[1]:
                mask = np.tile(mask, [1, cavity_cov.shape[1]])

        pseudo_y, pseudo_var = model.compute_full_pseudo_lik()
        lZ_pseudo = vmap(mvn_logpdf)(
            pseudo_y,
            cavity_mean,
            pseudo_var / self.power + cavity_cov,
            mask
        )
        constant = vmap(pep_constant, [0, None, 0])(pseudo_var, self.power, mask)  # PEP constant
        lZ_pseudo += constant

        lZ_post = model.compute_log_lik(pseudo_y, pseudo_var)

        ep_energy = -(
            lZ_post
            + 1 / self.power * (scale * np.nansum(lZ) - np.nansum(lZ_pseudo))
        )

        return ep_energy


class PosteriorLinearisation(Inference):
    """
    An iterated smoothing algorithm based on statistical linear regression (SLR).
    This method linearises the likelihood model in the region described by the posterior.
    """
    def __init__(self,
                 cubature=GaussHermite(),
                 energy_function=None):
        super().__init__(cubature=cubature)
        if energy_function is None:
            self.energy_function = ExpectationPropagation(power=1, cubature=cubature).energy
            # self.energy_function = VariationalInference(cubature=cubature).energy
        else:
            self.energy_function = energy_function
        self.name = 'Posterior Linearisation (PL)'

    def update(self, model, batch_ind=None, lr=1.):
        """
        """
        if batch_ind is None:
            batch_ind = np.arange(model.num_data)

        mean_f, cov_f = model.conditional_posterior_to_data(batch_ind)

        # PL expected density is mu=E_q[E(y|f)]
        mu, d_mu, omega = vmap(model.likelihood.statistical_linear_regression, (0, 0, None))(
            mean_f,
            cov_f,
            self.cubature
        )
        residual = model.Y[batch_ind].reshape(mu.shape) - mu
        mask = np.isnan(residual)
        residual = np.where(mask, 0., residual)

        dmu_omega = transpose(vmap(solve)(omega, d_mu))  # d_mu^T @ inv(omega)
        jacobian = dmu_omega @ residual
        hessian = -dmu_omega @ d_mu

        hessian = -ensure_diagonal_positive_precision(-hessian)  # manual fix to avoid non-PSD precision

        # deal with missing data
        jacobian = np.where(mask, np.nan, jacobian)
        hessian = np.where(diag(np.squeeze(mask, axis=-1)), np.nan, hessian)

        jacobian, hessian = model.conditional_data_to_posterior(jacobian, hessian)

        if mean_f.shape[1] == jacobian.shape[1]:
            return mean_f, jacobian, hessian
        else:
            ind = model.ind[batch_ind]
            return model.posterior_mean.value[ind], jacobian, hessian  # sparse Markov case

    def energy(self, model, batch_ind=None):
        """
        The PL energy given in [1] is a poor approximation to the EP energy (although the gradients are ok, since
        the part they discard does not depends on the hyperparameters). Therefore, we can choose to use either
        the variational free energy or EP energy here.
        TODO: develop a PL energy approximation that reuses the linearisation quantities and matches GHS / UKS etc.
        [1] Garcia-Fernandez, Tronarp, Särkkä (2018) 'Gaussian process classification using posterior linearisation'
        """
        return self.energy_function(model, batch_ind)


class Taylor(Inference):
    """
    Inference using analytical linearisation, i.e. a first order Taylor expansion. This is equivalent to
    the Extended Kalman Smoother when using a Markov GP.
    """
    def __init__(self,
                 cubature=GaussHermite(),  # cubature is only used in the energy calc. TODO: remove need for this
                 energy_function=None):
        super().__init__(cubature=cubature)
        self.name = 'Taylor / Extended Kalman Smoother (EKS)'
        if energy_function is None:
            self.energy_function = ExpectationPropagation(power=1, cubature=cubature).energy
            # self.energy_function = VariationalInference(cubature=cubature).energy
        else:
            self.energy_function = energy_function

    def update(self, model, batch_ind=None, lr=1.):
        """
        """
        if batch_ind is None:
            batch_ind = np.arange(model.num_data)
        Y = model.Y[batch_ind]

        mean_f, cov_f = model.conditional_posterior_to_data(batch_ind)

        # calculate the Jacobian of the observation model w.r.t. function fₙ and noise term rₙ
        Jf, Jsigma = vmap(model.likelihood.analytical_linearisation)(mean_f, np.zeros_like(Y))  # evaluate at mean

        obs_cov = np.eye(Y.shape[1])  # observation noise scale is w.l.o.g. 1
        sigma = Jsigma @ obs_cov @ transpose(Jsigma)
        likelihood_expectation, _ = vmap(model.likelihood.conditional_moments)(mean_f)
        residual = Y.reshape(likelihood_expectation.shape) - likelihood_expectation  # residual, yₙ-E[yₙ|fₙ]

        mask = np.isnan(residual)
        residual = np.where(mask, 0., residual)

        Jf_invsigma = transpose(vmap(solve)(sigma, Jf))  # Jf^T @ sigma
        jacobian = Jf_invsigma @ residual
        hessian = -Jf_invsigma @ Jf

        # deal with missing data
        jacobian = np.where(mask, np.nan, jacobian)
        hessian = np.where(diag(np.squeeze(mask, axis=-1)), np.nan, hessian)

        jacobian, hessian = model.conditional_data_to_posterior(jacobian, hessian)

        if mean_f.shape[1] == jacobian.shape[1]:
            return mean_f, jacobian, hessian
        else:
            ind = model.ind[batch_ind]
            return model.posterior_mean.value[ind], jacobian, hessian  # sparse Markov case

    def energy(self, model, batch_ind=None):
        """
        Arguably, we should use the filtering energy here, such that the result matches that of the standard
        extended Kalman smoother.
        TODO: implement energy that matches standard EKS
        """
        return self.energy_function(model, batch_ind)


class ExtendedKalmanSmoother(Taylor):
    pass
