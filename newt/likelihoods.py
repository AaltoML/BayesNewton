import objax
import jax.numpy as np
from jax import grad, jacrev, vmap
from jax.scipy.special import erf, gammaln, logsumexp
from jax.scipy.linalg import cholesky
from .cubature import (
    gauss_hermite,
    variational_expectation_cubature,
    moment_match_cubature,
    statistical_linear_regression_cubature,
    log_density_cubature,
    predict_cubature
)
from .utils import (
    solve,
    transpose,
    softplus,
    softplus_inv,
    sigmoid,
    sigmoid_diff,
    pep_constant,
    mvn_logpdf,
    mvn_logpdf_and_derivs
)
import math

LOG2PI = math.log(2 * math.pi)


class Likelihood(objax.Module):
    """
    The likelihood model class, p(yₙ|fₙ). Each likelihood implements its own methods used during inference:
        Moment matching is used for EP
        Variational expectation is used for VI
        Statistical linearisation is used for PL
        Analytical linearisation is used for EKS
        Log-likelihood gradients are used for Laplace
    If no custom parameter update method is provided, cubature is used (Gauss-Hermite by default).
    The requirement for all inference methods to work is the implementation of the following methods:
        evaluate_likelihood(), which simply evaluates the likelihood given the latent function
        evaluate_log_likelihood()
        conditional_moments(), which return E[y|f] and Cov[y|f]
    """

    def __call__(self, y, f):
        return self.evaluate_likelihood(y, f)

    def evaluate_likelihood(self, y, f):
        raise NotImplementedError

    def evaluate_log_likelihood(self, y, f):
        raise NotImplementedError

    def conditional_moments(self, f):
        raise NotImplementedError

    def log_likelihood_gradients_(self, y, f):
        """
        Evaluate the Jacobian and Hessian of the log-likelihood
        """
        log_lik = self.evaluate_log_likelihood(y, f)
        f = np.squeeze(f)
        J = jacrev(self.evaluate_log_likelihood, argnums=1)
        H = jacrev(J, argnums=1)
        return log_lik, J(y, f), H(y, f)

    def log_likelihood_gradients(self, y, f):
        """
        Most likelihoods factorise across data points. For multi-latent models, a custom method must be implemented.
        """
        # align shapes and compute mask
        y = y.reshape(-1, 1)
        f = f.reshape(-1, 1)
        mask = np.isnan(y)
        y = np.where(mask, f, y)

        # compute gradients of the log likelihood
        log_lik, J, H = vmap(self.log_likelihood_gradients_)(y, f)

        # apply mask
        mask = np.squeeze(mask)
        log_lik = np.where(mask, 0., log_lik)
        J = np.where(mask, np.nan, J)
        H = np.where(mask, np.nan, H)

        return log_lik, J, np.diag(H)

    def variational_expectation_(self, y, m, v, cubature=None):
        """
        If no custom variational expectation method is provided, we use cubature.
        """
        return variational_expectation_cubature(self, y, m, v, cubature)

    def variational_expectation(self, y, m, v, cubature=None):
        """
        Most likelihoods factorise across data points. For multi-latent models, a custom method must be implemented.
        """

        # align shapes and compute mask
        y = y.reshape(-1, 1, 1)
        m = m.reshape(-1, 1, 1)
        v = np.diag(v).reshape(-1, 1, 1)
        mask = np.isnan(y)
        y = np.where(mask, m, y)

        # compute variational expectations and their derivatives
        var_exp, dE_dm, d2E_dm2 = vmap(self.variational_expectation_, (0, 0, 0, None))(y, m, v, cubature)

        # apply mask
        var_exp = np.where(np.squeeze(mask), 0., np.squeeze(var_exp))
        dE_dm = np.where(mask, np.nan, dE_dm)
        d2E_dm2 = np.where(mask, np.nan, d2E_dm2)

        return var_exp, np.squeeze(dE_dm, axis=2), np.diag(np.squeeze(d2E_dm2, axis=(1, 2)))

    def log_density(self, y, mean, cov, cubature=None):
        """
        """
        return log_density_cubature(self, y, mean, cov, cubature)

    def moment_match_(self, y, cav_mean, cav_cov, power=1.0, cubature=None):
        """
        If no custom moment matching method is provided, we use cubature.
        """
        return moment_match_cubature(self, y, cav_mean, cav_cov, power, cubature)

    def moment_match(self, y, m, v, power=1.0, cubature=None):
        """
        """
        # align shapes and compute mask
        y = y.reshape(-1, 1)
        m = m.reshape(-1, 1)
        mask = np.isnan(y)
        y = np.where(mask, m, y)

        lZ, dlZ, d2lZ = self.moment_match_(y, m, v, power, cubature)

        return lZ, dlZ, d2lZ

    def statistical_linear_regression_(self, m, v, cubature=None):
        """
        If no custom SLR method is provided, we use cubature.
        """
        return statistical_linear_regression_cubature(self, m, v, cubature)

    def statistical_linear_regression(self, m, v, cubature=None):
        """
        Most likelihoods factorise across data points. For multi-latent models, a custom method must be implemented.
        TODO: multi-dim case
        """

        # align shapes and compute mask
        m = m.reshape(-1, 1, 1)
        v = np.diag(v).reshape(-1, 1, 1)

        # compute SLR
        mu, omega, d_mu, d2_mu = vmap(self.statistical_linear_regression_, (0, 0, None))(m, v, cubature)
        return (
            np.squeeze(mu, axis=2),
            np.diag(np.squeeze(omega, axis=(1, 2))),
            np.diag(np.squeeze(d_mu, axis=(1, 2))),
            np.diag(np.squeeze(d2_mu, axis=(1, 2))),
        )

    def observation_model(self, f, sigma):
        """
        The implicit observation model is:
            h(fₙ,rₙ) = E[yₙ|fₙ] + √Cov[yₙ|fₙ] σₙ
        """
        conditional_expectation, conditional_covariance = self.conditional_moments(f)
        obs_model = conditional_expectation + cholesky(conditional_covariance) @ sigma
        return np.squeeze(obs_model)

    def jac_obs(self, f, sigma):
        return np.squeeze(jacrev(self.observation_model, argnums=0)(f, sigma))

    def jac_obs_sigma(self, f, sigma):
        return np.squeeze(jacrev(self.observation_model, argnums=1)(f, sigma))

    def analytical_linearisation(self, m, sigma=None):
        """
        Compute the Jacobian of the state space observation model w.r.t. the
        function fₙ and the noise term σₙ.
        The implicit observation model is:
            h(fₙ,rₙ) = E[yₙ|fₙ] + √Cov[yₙ|fₙ] σₙ
        The Jacobians are evaluated at the means, fₙ=m, σₙ=0, to be used during
        Extended Kalman smoothing.
        """
        sigma = np.array([[0.0]]) if sigma is None else sigma

        m = m.reshape(-1, 1, 1)
        sigma = sigma.reshape(-1, 1, 1)

        Jf, Jsigma = vmap(jacrev(self.observation_model, argnums=(0, 1)))(m, sigma)

        Hf = vmap(jacrev(self.jac_obs, argnums=0))(m, sigma)
        Hsigma = vmap(jacrev(self.jac_obs_sigma, argnums=1))(m, sigma)

        return (
            np.diag(np.squeeze(Jf, axis=(1, 2))),
            np.diag(np.squeeze(Hf, axis=(1, 2))),
            np.diag(np.squeeze(Jsigma, axis=(1, 2))),
            np.diag(np.squeeze(Hsigma, axis=(1, 2))),
        )

    def predict(self, mean_f, var_f, cubature=None):
        """
        predict in data space given predictive mean and var of the latent function
        TODO: multi-latent case
        """
        if mean_f.shape[0] > 1:
            return vmap(predict_cubature, [None, 0, 0, None])(
                self,
                mean_f.reshape(-1, 1, 1),
                var_f.reshape(-1, 1, 1),
                cubature
            )
        else:
            return predict_cubature(self, mean_f, var_f, cubature)


class Gaussian(Likelihood):
    """
    The Gaussian likelihood:
        p(yₙ|fₙ) = 𝓝(yₙ|fₙ,σ²)
    """
    def __init__(self,
                 variance=0.1,
                 fix_variance=False):
        """
        :param variance: The observation noise variance, σ²
        """
        if fix_variance:
            self.transformed_variance = objax.StateVar(np.array(softplus_inv(variance)))
        else:
            self.transformed_variance = objax.TrainVar(np.array(softplus_inv(variance)))
        super().__init__()
        self.name = 'Gaussian'
        self.link_fn = lambda f: f

    @property
    def variance(self):
        return softplus(self.transformed_variance.value)

    def evaluate_likelihood(self, y, f):
        """
        Evaluate the Gaussian function 𝓝(yₙ|fₙ,σ²).
        Can be used to evaluate Q cubature points.
        :param y: observed data yₙ [scalar]
        :param f: mean, i.e. the latent function value fₙ [Q, 1]
        :return:
            𝓝(yₙ|fₙ,σ²), where σ² is the observation noise [Q, 1]
        """
        return (2 * np.pi * self.variance) ** -0.5 * np.exp(-0.5 * (y - f) ** 2 / self.variance)

    def evaluate_log_likelihood(self, y, f):
        """
        Evaluate the log-Gaussian function log𝓝(yₙ|fₙ,σ²).
        Can be used to evaluate Q cubature points.
        :param y: observed data yₙ [scalar]
        :param f: mean, i.e. the latent function value fₙ [Q, 1]
        :return:
            log𝓝(yₙ|fₙ,σ²), where σ² is the observation noise [Q, 1]
        """
        return np.squeeze(-0.5 * np.log(2 * np.pi * self.variance) - 0.5 * (y - f) ** 2 / self.variance)

    def conditional_moments(self, f):
        """
        The first two conditional moments of a Gaussian are the mean and variance:
            E[y|f] = f
            Var[y|f] = σ²
        """
        return f, np.array([[self.variance]])

    def variational_expectation_(self, y, post_mean, post_cov, cubature=None):
        """
        Computes the "variational expectation", i.e. the
        expected log-likelihood, and its derivatives w.r.t. the posterior mean
            E[log p(yₙ|fₙ)] = ∫ log p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        :param y: observed data (yₙ) [scalar]
        :param post_mean: posterior mean (mₙ) [scalar]
        :param post_cov: posterior variance (vₙ) [scalar]
        :param cubature: the function to compute sigma points and weights to use during cubature
        :return:
            exp_log_lik: the expected log likelihood, E[log p(yₙ|fₙ)]  [scalar]
            dE_dm: derivative of E[log p(yₙ|fₙ)] w.r.t. mₙ  [scalar]
            d2E_dm2: 2nd derivative of E[log p(yₙ|fₙ)] w.r.t. mₙ  [scalar]
        """
        # TODO: multi-dim case
        # Compute expected log likelihood:
        # E[log p(yₙ|fₙ)] = ∫ log p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        exp_log_lik = (
            -0.5 * np.log(2 * np.pi)
            - 0.5 * np.log(self.variance)
            - 0.5 * ((y - post_mean) ** 2 + post_cov) / self.variance
        )
        # Compute first derivative:
        dE_dm = (y - post_mean) / self.variance
        # Compute second derivative:
        d2E_dm2 = -1 / self.variance
        return exp_log_lik, dE_dm, d2E_dm2.reshape(-1, 1)

    def moment_match_(self, y, cav_mean, cav_cov, power=1.0, cubature=None):
        """
        Closed form Gaussian moment matching.
        Calculates the log partition function of the EP tilted distribution:
            logZₙ = log ∫ 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ = E[𝓝(yₙ|fₙ,σ²)]
        and its derivatives w.r.t. mₙ, which are required for moment matching.
        :param y: observed data (yₙ)
        :param cav_mean: cavity mean (mₙ)
        :param cav_cov: cavity covariance (vₙ)
        :param power: EP power [scalar]
        :param cubature: not used
        :return:
            lZ: the log partition function, logZₙ [scalar]
            dlZ: first derivative of logZₙ w.r.t. mₙ (if derivatives=True) [scalar]
            d2lZ: second derivative of logZₙ w.r.t. mₙ (if derivatives=True) [scalar]
        """
        lik_cov = self.variance * np.eye(cav_cov.shape[0])
        # log partition function, lZ:
        # logZₙ = log ∫ 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ
        #       = log 𝓝(yₙ|mₙ,σ²+vₙ)
        lZ, dlZ, d2lZ = mvn_logpdf_and_derivs(
            y,
            cav_mean,
            lik_cov / power + cav_cov
        )
        constant = pep_constant(lik_cov, power)
        lZ += constant
        return lZ, dlZ, d2lZ

    def log_density(self, y, mean, cov, cubature=None):
        """
        logZₙ = log ∫ 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ = E[𝓝(yₙ|fₙ,σ²)]
        :param y: observed data (yₙ)
        :param mean: cavity mean (mₙ)
        :param cov: cavity variance (vₙ)
        :param cubature: not used
        :return:
            lZ: the log density, logZₙ [scalar]
        """
        # logZₙ = log ∫ 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ = log 𝓝(yₙ|mₙ,σ²+vₙ)
        lZ = mvn_logpdf(
            y,
            mean,
            self.variance * np.eye(cov.shape[0]) + cov
        )
        return lZ

    def predict(self, mean_f, var_f, cubature=None):
        return mean_f, var_f + self.variance


class Bernoulli(Likelihood):
    """
    Bernoulli likelihood is p(yₙ|fₙ) = Pʸ(1-P)⁽¹⁻ʸ⁾, where P = E[yₙ=1|fₙ].
    Link function maps latent GP to [0,1].
    The Probit link function, i.e. the Error Function Likelihood:
        i.e. the Gaussian (Normal) cumulative density function:
        P = E[yₙ=1|fₙ] = Φ(fₙ)
                       = ∫ 𝓝(x|0,1) dx, where the integral is over (-∞, fₙ],
        The Normal CDF is calulcated using the error function:
                       = (1 + erf(fₙ / √2)) / 2
        for erf(z) = (2/√π) ∫ exp(-x²) dx, where the integral is over [0, z]
    The logit link function:
        P = E[yₙ=1|fₙ] = 1 / 1 + exp(-fₙ)
    """
    def __init__(self,
                 link='probit'):
        super().__init__()
        if link == 'logit':
            self.link_fn = lambda f: 1 / (1 + np.exp(-f))
            self.dlink_fn = lambda f: np.exp(f) / (1 + np.exp(f)) ** 2
            self.link = link
        elif link == 'probit':
            jitter = 1e-3
            self.link_fn = lambda f: 0.5 * (1.0 + erf(f / np.sqrt(2.0))) * (1 - 2 * jitter) + jitter
            self.dlink_fn = lambda f: grad(self.link_fn)(np.squeeze(f)).reshape(-1, 1)
            self.link = link
        else:
            raise NotImplementedError('link function not implemented')
        self.name = 'Bernoulli'

    def evaluate_likelihood(self, y, f):
        """
        :param y: observed data yₙ ϵ {-1, +1} [scalar]
        :param f: latent function value fₙ ϵ ℝ
        :return:
            p(yₙ|fₙ) = Pʸ(1-P)⁽¹⁻ʸ⁾
        """
        return np.where(np.equal(y, 1), self.link_fn(f), 1 - self.link_fn(f))

    def evaluate_log_likelihood(self, y, f):
        """
        :param y: observed data yₙ ϵ {-1, +1} [scalar]
        :param f: latent function value fₙ ϵ ℝ
        :return:
            log p(yₙ|fₙ)
        """
        return np.squeeze(np.log(self.evaluate_likelihood(y, f)))

    def conditional_moments(self, f):
        """
        The first two conditional moments of a Probit likelihood are:
            E[yₙ|fₙ] = Φ(fₙ)
            Var[yₙ|fₙ] = Φ(fₙ) (1 - Φ(fₙ))
        """
        return self.link_fn(f), self.link_fn(f)-(self.link_fn(f)**2)


class Probit(Bernoulli):
    """
    The probit likelihood = Bernoulli likelihood with probit link.
    """
    def __init__(self):
        super().__init__(link='probit')


class Erf(Probit):
    """
    The error function likelihood = probit = Bernoulli likelihood with probit link.
    """
    pass


class Logit(Bernoulli):
    """
    The logit likelihood = Bernoulli likelihood with logit link.
    """
    def __init__(self):
        super().__init__(link='logit')


class Logistic(Logit):
    """
    The logistic likelihood = logit = Bernoulli likelihood with logit link.
    """
    pass


class Poisson(Likelihood):
    """
    TODO: tidy docstring
    The Poisson likelihood:
        p(yₙ|fₙ) = Poisson(fₙ) = μʸ exp(-μ) / yₙ!
    where μ = g(fₙ) = mean = variance is the Poisson intensity.
    yₙ is non-negative integer count data.
    No closed form moment matching is available, se we default to using cubature.

    Letting Zy = gamma(yₙ+1) = yₙ!, we get log p(yₙ|fₙ) = log(g(fₙ))yₙ - g(fₙ) - log(Zy)
    The larger the intensity μ, the stronger the likelihood resembles a Gaussian
    since skewness = 1/sqrt(μ) and kurtosis = 1/μ.
    Two possible link functions:
    'exp':      link(fₙ) = exp(fₙ),         we have p(yₙ|fₙ) = exp(fₙyₙ-exp(fₙ))           / Zy.
    'logistic': link(fₙ) = log(1+exp(fₙ))), we have p(yₙ|fₙ) = logʸ(1+exp(fₙ)))(1+exp(fₙ)) / Zy.
    """
    def __init__(self,
                 binsize=1,
                 link='exp'):
        """
        :param link: link function, either 'exp' or 'logistic'
        """
        super().__init__()
        if link == 'exp':
            self.link_fn = np.exp
            self.dlink_fn = np.exp
            self.d2link_fn = np.exp
        elif link == 'logistic':
            self.link_fn = softplus
            self.dlink_fn = sigmoid
            self.d2link_fn = sigmoid_diff
        else:
            raise NotImplementedError('link function not implemented')
        self.binsize = np.array(binsize)
        self.name = 'Poisson'

    def evaluate_likelihood(self, y, f):
        """
        Evaluate the Poisson likelihood:
            p(yₙ|fₙ) = Poisson(fₙ) = μʸ exp(-μ) / yₙ!
        for μ = g(fₙ), where g() is the link function (exponential or logistic).
        We use the gamma function to evaluate yₙ! = gamma(yₙ + 1).
        Can be used to evaluate Q cubature points when performing moment matching.
        :param y: observed data (yₙ) [scalar]
        :param f: latent function value (fₙ) [Q, 1]
        :return:
            Poisson(fₙ) = μʸ exp(-μ) / yₙ! [Q, 1]
        """
        mu = self.link_fn(f) * self.binsize
        return np.exp(y * np.log(mu) - mu - gammaln(y + 1))

    def evaluate_log_likelihood(self, y, f):
        """
        Evaluate the Poisson log-likelihood:
            log p(yₙ|fₙ) = log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!)
        for μ = g(fₙ), where g() is the link function (exponential or logistic).
        We use the gamma function to evaluate yₙ! = gamma(yₙ + 1).
        Can be used to evaluate Q cubature points when performing moment matching.
        :param y: observed data (yₙ) [scalar]
        :param f: latent function value (fₙ) [Q, 1]
        :return:
            log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!) [Q, 1]
        """
        mu = self.link_fn(f) * self.binsize
        return np.squeeze(y * np.log(mu) - mu - gammaln(y + 1))

    def observation_model(self, f, sigma):
        """
        TODO: sort out broadcasting so we don't need this additional function (only difference is the transpose)
        The implicit observation model is:
            h(fₙ,rₙ) = E[yₙ|fₙ] + √Cov[yₙ|fₙ] σₙ
        """
        conditional_expectation, conditional_covariance = self.conditional_moments(f)
        obs_model = conditional_expectation + cholesky(conditional_covariance.T) @ sigma
        return np.squeeze(obs_model)

    def conditional_moments(self, f):
        """
        The first two conditional moments of a Poisson distribution are equal to the intensity:
            E[yₙ|fₙ] = link(fₙ)
            Var[yₙ|fₙ] = link(fₙ)
        """
        # TODO: multi-dim case
        return self.link_fn(f) * self.binsize, self.link_fn(f) * self.binsize
        # return self.link_fn(f) * self.binsize, vmap(np.diag, 1, 2)(self.link_fn(f) * self.binsize)

    def analytical_linearisation(self, m, sigma=None):
        """
        Compute the Jacobian of the state space observation model w.r.t. the
        function fₙ and the noise term σₙ.
        """
        link_fm = self.link_fn(m) * self.binsize
        dlink_fm = self.dlink_fn(m) * self.binsize
        d2link_fm = self.d2link_fn(m) * self.binsize
        Jf = np.diag(np.squeeze(dlink_fm + 0.5 * link_fm ** -0.5 * dlink_fm * sigma.reshape(-1, 1), axis=-1))
        Hf = np.diag(np.squeeze(d2link_fm
                                - 0.25 * link_fm ** -1.5 * dlink_fm ** 2 * sigma.reshape(-1, 1)
                                + 0.5 * link_fm ** -0.5 * d2link_fm * sigma.reshape(-1, 1)
                                , axis=-1))
        Jsigma = np.diag(np.squeeze(link_fm ** 0.5, axis=-1))
        Hsigma = np.zeros_like(Jsigma)
        return Jf, Hf, Jsigma, Hsigma

    def variational_expectation_(self, y, post_mean, post_cov, cubature=None):
        """
        Computes the "variational expectation", i.e. the
        expected log-likelihood, and its derivatives w.r.t. the posterior mean
        Let a = E[f] = m and b = E[exp(f)] = exp(m+v/2) then
        E[log Poisson(y | exp(f)*binsize)] = Y log binsize  + E[Y * log exp(f)] - E[binsize * exp(f)] - log Y!
                                           = Y log binsize + Y * m - binsize * exp(m + v/2) - log Y!
        :param y: observed data (yₙ) [scalar]
        :param post_mean: posterior mean (mₙ) [scalar]
        :param post_cov: posterior variance (vₙ) [scalar]
        :param cubature: the function to compute sigma points and weights to use during cubature
        :return:
            exp_log_lik: the expected log likelihood, E[log p(yₙ|fₙ)]  [scalar]
            dE_dm: derivative of E[log p(yₙ|fₙ)] w.r.t. mₙ  [scalar]
            d2E_dm2: 2nd derivative of E[log p(yₙ|fₙ)] w.r.t. mₙ  [scalar]
        """
        # TODO: multi-dim case
        exp_mean_cov = self.binsize * np.exp(post_mean + post_cov / 2)
        # Compute expected log likelihood:
        exp_log_lik = (
            y * np.log(self.binsize)
            + y * post_mean
            - exp_mean_cov
            - gammaln(y + 1.0)
        )
        # Compute first derivative:
        dE_dm = y - exp_mean_cov
        # Compute second derivative:
        d2E_dm2 = -exp_mean_cov
        return exp_log_lik, dE_dm, d2E_dm2.reshape(-1, 1)


def negative_binomial(m, y, alpha):
    k = 1 / alpha
    return (
        gammaln(k + y)
        - gammaln(y + 1)
        - gammaln(k)
        + y * np.log(m / (m + k))
        - k * np.log(1 + m * alpha)
    )


class NegativeBinomial(Likelihood):
    """
    BinTayyash et. al. 2021: Non-Parametric Modelling of Temporal and Spatial Counts Data From RNA-SEQ Experiments
    """
    def __init__(self,
                 alpha=1.0,
                 link='exp',
                 scale=1.0):
        """
        :param link: link function, either 'exp' or 'logistic'
        """
        super().__init__()
        if link == 'exp':
            self.link_fn = lambda mu: np.exp(mu)
            self.dlink_fn = lambda mu: np.exp(mu)
        elif link == 'logistic':
            self.link_fn = lambda mu: softplus(mu)
            self.dlink_fn = lambda mu: sigmoid(mu)
        else:
            raise NotImplementedError('link function not implemented')
        self.transformed_alpha = objax.TrainVar(np.array(softplus_inv(alpha)))
        self.scale = np.array(scale)
        self.name = 'Negative Binomial'

    @property
    def alpha(self):
        return softplus(self.transformed_alpha.value)

    def evaluate_likelihood(self, y, f):
        """
        """
        return np.exp(self.evaluate_log_likelihood(y, f))

    def evaluate_log_likelihood(self, y, f):
        """
        """
        return negative_binomial(self.link_fn(f) * self.scale, y, self.alpha)

    def conditional_moments(self, f):
        """
        """
        conditional_expectation = self.link_fn(f) * self.scale
        conditional_covariance = conditional_expectation + conditional_expectation ** 2 * self.alpha
        return conditional_expectation, conditional_covariance


class ZeroInflatedNegativeBinomial(Likelihood):
    """
    BinTayyash et. al. 2021: Non-Parametric Modelling of Temporal and Spatial Counts Data From RNA-SEQ Experiments
    """
    def __init__(self,
                 alpha=1.0,
                 km=1.0,
                 link='exp'):
        """
        :param link: link function, either 'exp' or 'logistic'
        """
        super().__init__()
        if link == 'exp':
            self.link_fn = lambda mu: np.exp(mu)
            self.dlink_fn = lambda mu: np.exp(mu)
        elif link == 'logistic':
            self.link_fn = lambda mu: softplus(mu)
            self.dlink_fn = lambda mu: sigmoid(mu)
        else:
            raise NotImplementedError('link function not implemented')
        self.transformed_alpha = objax.TrainVar(np.array(softplus_inv(alpha)))
        self.transformed_km = objax.TrainVar(np.array(km))
        self.name = 'Negative Binomial'

    @property
    def alpha(self):
        return softplus(self.transformed_alpha.value)

    @property
    def km(self):
        return softplus(self.transformed_km.value)

    def evaluate_likelihood(self, y, f):
        """
        """
        return np.exp(self.evaluate_log_likelihood(y, f))

    def evaluate_log_likelihood(self, y, f):
        """
        """
        m = self.link_fn(f)
        psi = 1. - (m / (self.km + m))
        nb_zero = - np.log(1. + m * self.alpha) / self.alpha
        log_p_zero = logsumexp(np.array([np.log(psi), np.log(1. - psi) + nb_zero]), axis=0)
        log_p_nonzero = np.log(1. - psi) + negative_binomial(m, y, self.alpha)
        return np.where(y == 0, log_p_zero, log_p_nonzero)

    def conditional_moments(self, f):
        """
        """
        m = self.link_fn(f)
        psi = 1. - (m / (self.km + m))
        conditional_expectation = m * (1 - psi)
        conditional_covariance = conditional_expectation * (1 + m * (psi + self.alpha))
        return conditional_expectation, conditional_covariance


class HeteroscedasticNoise(Likelihood):
    """
    The Heteroscedastic Noise likelihood:
        p(y|f1,f2) = N(y|f1,link(f2)^2)
    """
    def __init__(self, link='softplus'):
        """
        :param link: link function, either 'exp' or 'softplus' (note that the link is modified with an offset)
        """
        super().__init__()
        if link == 'exp':
            self.link_fn = np.exp
            self.dlink_fn = np.exp
            self.d2link_fn = np.exp
        elif link == 'softplus':
            self.link_fn = softplus
            self.dlink_fn = sigmoid
            self.d2link_fn = sigmoid_diff
        else:
            raise NotImplementedError('link function not implemented')
        self.name = 'Heteroscedastic Noise'

    def evaluate_likelihood(self, y, f):
        """
        Evaluate the likelihood
        """
        mu, var = self.conditional_moments(f)
        return (2 * np.pi * var) ** -0.5 * np.exp(-0.5 * (y - mu) ** 2 / var)

    def evaluate_log_likelihood(self, y, f):
        """
        Evaluate the log-likelihood
        """
        mu, var = self.conditional_moments(f)
        return np.squeeze(-0.5 * np.log(2 * np.pi * var) - 0.5 * (y - mu) ** 2 / var)

    def conditional_moments(self, f, hyp=None):
        """
        """
        return f[:1], self.link_fn(f[1:2]) ** 2

    def log_likelihood_gradients(self, y, f):
        log_lik, J, H = self.log_likelihood_gradients_(y, f)
        # H = -ensure_positive_precision(-H)
        return log_lik, J, H

    def log_density(self, y, mean, cov, cubature=None):
        """
        """
        if cubature is None:
            x, w = gauss_hermite(1, 20)  # Gauss-Hermite sigma points and weights
        else:
            x, w = cubature(1)
        # sigma_points = np.sqrt(2) * np.sqrt(v) * x + m  # scale locations according to cavity dist.
        sigma_points = np.sqrt(cov[1, 1]) * x + mean[1]  # fsigᵢ=xᵢ√cₙ + mₙ: scale locations according to cavity
        f2 = self.link_fn(sigma_points) ** 2.
        obs_var = f2 + cov[0, 0]
        normpdf = (2 * np.pi * obs_var) ** -0.5 * np.exp(-0.5 * (y - mean[0, 0]) ** 2 / obs_var)
        Z = np.sum(w * normpdf)
        lZ = np.log(np.maximum(Z, 1e-8))
        return lZ

    def moment_match__(self, y, cav_mean, cav_cov, power=1.0, cubature=None):
        """
        TODO: implement proper Hessian approx., as done in variational_expectation()
        """
        if cubature is None:
            x, w = gauss_hermite(1, 20)  # Gauss-Hermite sigma points and weights
        else:
            x, w = cubature(1)
        # sigma_points = np.sqrt(2) * np.sqrt(v) * x + m  # scale locations according to cavity dist.
        sigma_points = np.sqrt(cav_cov[1, 1]) * x + cav_mean[1]  # fsigᵢ=xᵢ√cₙ + mₙ: scale locations according to cavity

        f2 = self.link_fn(sigma_points) ** 2. / power
        obs_var = f2 + cav_cov[0, 0]
        const = power ** -0.5 * (2 * np.pi * self.link_fn(sigma_points) ** 2.) ** (0.5 - 0.5 * power)
        normpdf = const * (2 * np.pi * obs_var) ** -0.5 * np.exp(-0.5 * (y - cav_mean[0, 0]) ** 2 / obs_var)
        Z = np.sum(w * normpdf)
        Zinv = 1. / np.maximum(Z, 1e-8)
        lZ = np.log(np.maximum(Z, 1e-8))

        dZ_integrand1 = (y - cav_mean[0, 0]) / obs_var * normpdf
        dlZ1 = Zinv * np.sum(w * dZ_integrand1)

        dZ_integrand2 = (sigma_points - cav_mean[1, 0]) / cav_cov[1, 1] * normpdf
        dlZ2 = Zinv * np.sum(w * dZ_integrand2)

        d2Z_integrand1 = (-(f2 + cav_cov[0, 0]) ** -1 + ((y - cav_mean[0, 0]) / obs_var) ** 2) * normpdf
        d2lZ1 = -dlZ1 ** 2 + Zinv * np.sum(w * d2Z_integrand1)

        d2Z_integrand2 = (-cav_cov[1, 1] ** -1 + ((sigma_points - cav_mean[1, 0]) / cav_cov[1, 1]) ** 2) * normpdf
        d2lZ2 = -dlZ2 ** 2 + Zinv * np.sum(w * d2Z_integrand2)

        dlZ = np.block([[dlZ1],
                        [dlZ2]])
        d2lZ = np.block([[d2lZ1, 0],
                         [0., d2lZ2]])

        return lZ, dlZ, d2lZ

    def log_density_power(self, y, cav_mean, cav_cov, power=1.0, cubature=None):
        """
        """
        if cubature is None:
            x, w = gauss_hermite(1, 20)  # Gauss-Hermite sigma points and weights
        else:
            x, w = cubature(1)
        # sigma_points = np.sqrt(2) * np.sqrt(v) * x + m  # scale locations according to cavity dist.
        sigma_points = np.sqrt(cav_cov[1, 1]) * x + cav_mean[1]  # fsigᵢ=xᵢ√cₙ + mₙ: scale locations according to cavity

        f2 = self.link_fn(sigma_points) ** 2. / power
        obs_var = f2 + cav_cov[0, 0]
        const = power ** -0.5 * (2 * np.pi * self.link_fn(sigma_points) ** 2.) ** (0.5 - 0.5 * power)
        normpdf = const * (2 * np.pi * obs_var) ** -0.5 * np.exp(-0.5 * (y - cav_mean[0, 0]) ** 2 / obs_var)
        Z = np.sum(w * normpdf)
        lZ = np.log(np.maximum(Z, 1e-8))
        return lZ

    def log_density_dm(self, y, cav_mean, cav_cov, power=1.0, cubature=None):
        """
        """
        dE_dm = grad(self.log_density_power, argnums=1)(y, cav_mean, cav_cov, power, cubature)
        return dE_dm

    def log_density_dm2(self, y, cav_mean, cav_cov, power=1.0, cubature=None):
        """
        """
        d2E_dm2 = jacrev(self.log_density_dm, argnums=1)(y, cav_mean, cav_cov, power, cubature)
        return np.squeeze(d2E_dm2)

    def moment_match(self, y, cav_mean, cav_cov, power=1.0, cubature=None):
        """
        """
        E = self.log_density_power(y, cav_mean, cav_cov, power, cubature)
        dE_dm = self.log_density_dm(y, cav_mean, cav_cov, power, cubature)
        d2E_dm2 = self.log_density_dm2(y, cav_mean, cav_cov, power, cubature)
        # a, b, c = self.moment_match__(y, cav_mean, cav_cov, power, cubature)
        return E, dE_dm, d2E_dm2

    def log_expected_likelihood(self, y, x, w, cav_mean, cav_var, power):
        sigma_points = np.sqrt(cav_var[1]) * x + cav_mean[1]
        f2 = self.link_fn(sigma_points) ** 2. / power
        obs_var = f2 + cav_var[0]
        const = power ** -0.5 * (2 * np.pi * self.link_fn(sigma_points) ** 2.) ** (0.5 - 0.5 * power)
        normpdf = const * (2 * np.pi * obs_var) ** -0.5 * np.exp(-0.5 * (y - cav_mean[0]) ** 2 / obs_var)
        Z = np.sum(w * normpdf)
        lZ = np.log(Z + 1e-8)
        return lZ

    def expected_log_likelihood(self, y, m, v, cubature=None):
        """
        """
        if cubature is None:
            x, w = gauss_hermite(2, 20)  # Gauss-Hermite sigma points and weights
        else:
            x, w = cubature(2)
        # v = (v + v.T) / 2
        sigma_points = cholesky(v) @ x + m  # fsigᵢ=xᵢ√(2vₙ) + mₙ: scale locations according to cavity dist.
        # Compute expected log likelihood via cubature:
        # E[log p(yₙ|fₙ)] = ∫ log p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        #                 ≈ ∑ᵢ wᵢ log p(yₙ|fsigᵢ)
        exp_log_lik = np.sum(w * self.evaluate_log_likelihood(y, sigma_points))
        return exp_log_lik

    def expected_log_likelihood_dm(self, y, m, v, cubature=None):
        """
        """
        dE_dm = grad(self.expected_log_likelihood, argnums=1)(y, m, v, cubature)
        return dE_dm

    def expected_log_likelihood_dm2(self, y, m, v, cubature=None):
        """
        """
        d2E_dm2 = jacrev(self.expected_log_likelihood_dm, argnums=1)(y, m, v, cubature)
        return np.squeeze(d2E_dm2)

    def variational_expectation(self, y, m, v, cubature=None):
        """
        Compute expected log likelihood via cubature:
        E[log p(yₙ|fₙ)] = ∫ log p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        """
        E = self.expected_log_likelihood(y, m, v, cubature)
        dE_dm = self.expected_log_likelihood_dm(y, m, v, cubature)
        d2E_dm2 = self.expected_log_likelihood_dm2(y, m, v, cubature)
        # d2E_dm2 = -ensure_positive_precision(-d2E_dm2)
        # return E, dE_dm, np.diag(np.diag(d2E_dm2))  # TODO: check this is the same as above
        return E, dE_dm, d2E_dm2

    def statistical_linear_regression(self, mean, cov, cubature=None):
        """
        Perform statistical linear regression (SLR) using cubature.
        We aim to find a likelihood approximation p(yₙ|fₙ) ≈ 𝓝(yₙ|Afₙ+b,Ω+Var[yₙ|fₙ]).
        """
        if cubature is None:
            x, w = gauss_hermite(mean.shape[0], 20)  # Gauss-Hermite sigma points and weights
        else:
            x, w = cubature(mean.shape[0])
        m0, m1, v0, v1 = mean[0, 0], mean[1, 0], cov[0, 0], cov[1, 1]
        # fsigᵢ=xᵢ√(vₙ) + mₙ: scale locations according to cavity dist.
        sigma_points = cholesky(cov) @ x + mean
        var = self.link_fn(sigma_points[1]) ** 2
        # Compute muₙ via cubature:
        # muₙ = ∫ E[yₙ|fₙ] 𝓝(fₙ|mₙ,vₙ) dfₙ
        #    ≈ ∑ᵢ wᵢ E[yₙ|fsigᵢ]
        mu = m0.reshape(1, 1)
        # Compute variance S via cubature:
        # S = ∫ [(E[yₙ|fₙ]-muₙ) (E[yₙ|fₙ]-muₙ)' + Cov[yₙ|fₙ]] 𝓝(fₙ|mₙ,vₙ) dfₙ
        #   ≈ ∑ᵢ wᵢ [(E[yₙ|fsigᵢ]-muₙ) (E[yₙ|fsigᵢ]-muₙ)' + Cov[yₙ|fₙ]]
        S = v0 + np.sum(
            w * var
        )
        S = S.reshape(1, 1)
        # Compute cross covariance C via cubature:
        # C = ∫ (fₙ-mₙ) (E[yₙ|fₙ]-muₙ)' 𝓝(fₙ|mₙ,vₙ) dfₙ
        #   ≈ ∑ᵢ wᵢ (fsigᵢ -mₙ) (E[yₙ|fsigᵢ]-muₙ)'
        C = np.sum(
            w * (sigma_points - mean) * (sigma_points[0] - m0), axis=-1
        ).reshape(2, 1)
        # Compute derivative of z via cubature:
        # d_mu = ∫ E[yₙ|fₙ] vₙ⁻¹ (fₙ-mₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        #      ≈ ∑ᵢ wᵢ E[yₙ|fsigᵢ] vₙ⁻¹ (fsigᵢ-mₙ)
        d_mu = np.block([[1., 0.]])
        omega = S - transpose(C) @ solve(cov, C)
        d2_mu = np.nan  # TODO: IMPLEMENT
        return mu, omega, d_mu, d2_mu

    # def analytical_linearisation(self, m, sigma=None):
    #     """
    #     Compute the Jacobian of the state space observation model w.r.t. the
    #     function fₙ and the noise term σₙ.
    #     """
    #     Jf = np.block([[np.array(1.0), self.dlink_fn(m[1]) * sigma]])
    #     Hf = np.block([[np.array(0.0), np.array(0.0)],
    #                    [np.array(0.0), self.d2link_fn(m[1]) * sigma]])
    #     Jsigma = self.link_fn(np.array([m[1]]))
    #     Hsigma = np.zeros_like(Jsigma)
    #     return Jf, Hf, Jsigma, Hsigma

    def analytical_linearisation(self, m, sigma=None):
        """
        Compute the Jacobian of the state space observation model w.r.t. the
        function fₙ and the noise term σₙ.
        The implicit observation model is:
            h(fₙ,rₙ) = E[yₙ|fₙ] + √Cov[yₙ|fₙ] σₙ
        The Jacobians are evaluated at the means, fₙ=m, σₙ=0, to be used during
        Extended Kalman smoothing.
        """
        sigma = np.array([[0.0]]) if sigma is None else sigma

        Jf, Jsigma = jacrev(self.observation_model, argnums=(0, 1))(m, sigma)

        Hf = jacrev(self.jac_obs, argnums=0)(m, sigma)
        Hsigma = jacrev(self.jac_obs_sigma, argnums=1)(m, sigma)

        return Jf.T, np.swapaxes(Hf, axis1=0, axis2=2), Jsigma[None], Hsigma[None]


class NonnegativeMatrixFactorisation(Likelihood):
    """
    The Nonnegative Matrix Factorisation likelihood
    """
    def __init__(self, num_subbands, num_modulators, variance=0.1, weights=None):
        """
        param hyp: observation noise
        """
        self.transformed_variance = objax.TrainVar(np.array(softplus_inv(variance)))
        super().__init__()
        self.name = 'Nonnegative Matrix Factorisation'
        self.link_fn = softplus
        self.dlink_fn = sigmoid  # derivative of the link function
        self.d2link_fn = sigmoid_diff   # 2nd derivative of the link function
        self.num_subbands = num_subbands
        self.num_modulators = num_modulators
        if weights is None:
            weights = objax.random.uniform(shape=(num_subbands, num_modulators))
        self.transformed_weights = objax.TrainVar(softplus_inv(weights))

    @property
    def variance(self):
        return softplus(self.transformed_variance.value)

    @property
    def weights(self):
        return softplus(self.transformed_weights.value)

    def evaluate_likelihood(self, y, f):
        """
        Evaluate the likelihood
        """
        mu, var = self.conditional_moments(f)
        return (2 * np.pi * var) ** -0.5 * np.exp(-0.5 * (y - mu) ** 2 / var)

    def evaluate_log_likelihood(self, y, f):
        """
        Evaluate the log-likelihood
        """
        mu, var = self.conditional_moments(f)
        return np.squeeze(-0.5 * np.log(2 * np.pi * var) - 0.5 * (y - mu) ** 2 / var)

    def conditional_moments(self, f):
        """
        """
        subbands, modulators = f[:self.num_subbands], self.link_fn(f[self.num_subbands:])
        return np.sum(subbands * (self.weights @ modulators)).reshape(-1, 1), np.array([[self.variance]])
        # return np.atleast_2d(modulators.T @ subbands),  np.atleast_2d(obs_noise_var)

    def log_likelihood_gradients(self, y, f):
        log_lik, J, H = self.log_likelihood_gradients_(y, f)
        return log_lik, J, H

    def log_density(self, y, mean, cov, cubature=None):
        """
        """
        if cubature is None:
            x, w = gauss_hermite(self.num_modulators, 20)  # Gauss-Hermite sigma points and weights
        else:
            x, w = cubature(self.num_modulators)

        # subband_mean, modulator_mean = mean[:self.num_subbands], self.link_fn(mean[self.num_subbands:])
        subband_mean, modulator_mean = mean[:self.num_subbands], mean[self.num_subbands:]  # TODO: CHECK
        subband_cov = cov[:self.num_subbands, :self.num_subbands]
        modulator_cov = cov[self.num_subbands:, self.num_subbands:]
        subband_var = np.diag(subband_cov)[..., None]
        sigma_points = cholesky(modulator_cov) @ x + modulator_mean
        modulator_mean_positive = self.weights @ self.link_fn(sigma_points)
        mu = (modulator_mean_positive.T @ subband_mean)[:, 0]
        var = self.variance + (modulator_mean_positive.T ** 2 @ subband_var)[:, 0]
        normpdf = (2 * np.pi * var) ** -0.5 * np.exp(-0.5 * (y - mu) ** 2 / var)
        Z = np.sum(w * normpdf)
        lZ = np.log(Z + 1e-8)
        return lZ

    def log_density_power(self, y, cav_mean, cav_cov, power=1.0, cubature=None):
        """
        """
        if cubature is None:
            x, w = gauss_hermite(self.num_modulators, 20)  # Gauss-Hermite sigma points and weights
        else:
            x, w = cubature(self.num_modulators)

        subband_mean, modulator_mean = cav_mean[:self.num_subbands], cav_mean[self.num_subbands:]  # TODO: CHECK
        subband_cov = cav_cov[:self.num_subbands, :self.num_subbands]
        modulator_cov = cav_cov[self.num_subbands:, self.num_subbands:]
        subband_var = np.diag(subband_cov)[..., None]
        sigma_points = cholesky(modulator_cov) @ x + modulator_mean
        modulator_mean_positive = self.weights @ self.link_fn(sigma_points)
        const = power ** -0.5 * (2 * np.pi * self.variance) ** (0.5 - 0.5 * power)
        mu = (modulator_mean_positive.T @ subband_mean)[:, 0]
        var = self.variance / power + (modulator_mean_positive.T ** 2 @ subband_var)[:, 0]
        normpdf = const * (2 * np.pi * var) ** -0.5 * np.exp(-0.5 * (y - mu) ** 2 / var)
        Z = np.sum(w * normpdf)
        lZ = np.log(Z + 1e-8)
        return lZ

    def log_density_dm(self, y, cav_mean, cav_cov, power=1.0, cubature=None):
        """
        """
        dE_dm = grad(self.log_density_power, argnums=1)(y, cav_mean, cav_cov, power, cubature)
        return dE_dm

    def log_density_dm2(self, y, cav_mean, cav_cov, power=1.0, cubature=None):
        """
        """
        d2E_dm2 = jacrev(self.log_density_dm, argnums=1)(y, cav_mean, cav_cov, power, cubature)
        return np.squeeze(d2E_dm2)

    def moment_match(self, y, cav_mean, cav_cov, power=1.0, cubature=None):
        """
        """
        E = self.log_density_power(y, cav_mean, cav_cov, power, cubature)
        dE_dm = self.log_density_dm(y, cav_mean, cav_cov, power, cubature)
        d2E_dm2 = self.log_density_dm2(y, cav_mean, cav_cov, power, cubature)
        return E, dE_dm, d2E_dm2

    def expected_conditional_mean(self, mean, cov, cubature=None):
        """
        Compute Eq[E[y|f]] = ∫ Ey[p(y|f)] 𝓝(f|mean,cov) dfₙ
        TODO: this needs checking - not sure the weights have been applied correctly
        """
        if cubature is None:
            x, w = gauss_hermite(self.num_modulators, 20)  # Gauss-Hermite sigma points and weights
        else:
            x, w = cubature(self.num_modulators)

        # subband_mean, modulator_mean = mean[:num_components], self.link_fn(mean[num_components:])
        subband_mean, modulator_mean = mean[:self.num_subbands], mean[self.num_subbands:]  # TODO: CHECK
        subband_cov = cov[:self.num_subbands, :self.num_subbands]
        modulator_cov = cov[self.num_subbands:, self.num_subbands:]
        subband_var = np.diag(subband_cov)[..., None]

        sigma_points = cholesky(modulator_cov) @ x + modulator_mean
        modulator_mean_positive = self.weights @ self.link_fn(sigma_points)
        lik_expectation, lik_covariance = (modulator_mean_positive.T @ subband_mean).T, self.variance
        # Compute muₙ via cubature:
        # muₙ = ∫ E[yₙ|fₙ] 𝓝(fₙ|mₙ,vₙ) dfₙ
        #    ≈ ∑ᵢ wᵢ E[yₙ|fsigᵢ]
        mu = np.sum(
            w * lik_expectation, axis=-1
        )[:, None]
        # Compute variance S via cubature:
        # S = ∫ [(E[yₙ|fₙ]-muₙ) (E[yₙ|fₙ]-muₙ)' + Cov[yₙ|fₙ]] 𝓝(fₙ|mₙ,vₙ) dfₙ
        #   ≈ ∑ᵢ wᵢ [(E[yₙ|fsigᵢ]-muₙ) (E[yₙ|fsigᵢ]-muₙ)' + Cov[yₙ|fₙ]]
        S = np.sum(
            w * ((lik_expectation - mu) * (lik_expectation - mu) + lik_covariance), axis=-1
        )[:, None]
        # Compute cross covariance C via cubature:
        # C = ∫ (fₙ-mₙ) (E[yₙ|fₙ]-muₙ)' 𝓝(fₙ|mₙ,vₙ) dfₙ
        #   ≈ ∑ᵢ wᵢ (fsigᵢ -mₙ) (E[yₙ|fsigᵢ]-muₙ)'
        C = np.sum(
            w * np.block([[modulator_mean_positive * subband_var],
                          [sigma_points - modulator_mean]]) * (lik_expectation - mu), axis=-1
        )[:, None]
        # compute equivalent likelihood noise, omega
        omega = S - C.T @ solve(cov, C)
        return np.squeeze(mu), omega

    def expected_conditional_mean_dm(self, mean, cov, cubature=None):
        """
        """
        dmu_dm, _ = grad(self.expected_conditional_mean, argnums=0, has_aux=True)(mean, cov, cubature)
        return np.squeeze(dmu_dm)

    def expected_conditional_mean_dm2(self, mean, cov, cubature=None):
        """
        """
        d2mu_dm2 = jacrev(self.expected_conditional_mean_dm, argnums=0)(mean, cov, cubature)
        return d2mu_dm2

    def statistical_linear_regression(self, mean, cov, cubature=None):
        mu, omega = self.expected_conditional_mean(mean, cov, cubature)
        dmu_dm = self.expected_conditional_mean_dm(mean, cov, cubature)
        d2mu_dm2 = self.expected_conditional_mean_dm2(mean, cov, cubature)
        # return mu.reshape(-1, 1), omega, dmu_dm[None], d2mu_dm2[None]
        return mu.reshape(-1, 1), omega, dmu_dm[None], np.swapaxes(d2mu_dm2, axis1=0, axis2=2)

    def expected_log_likelihood(self, y, post_mean, post_cov, cubature=None):
        """
        """
        if cubature is None:
            x, w = gauss_hermite(self.num_modulators, 20)  # Gauss-Hermite sigma points and weights
        else:
            x, w = cubature(self.num_modulators)

        # subband_mean, modulator_mean = post_mean[:self.num_subbands], self.link_fn(post_mean[self.num_subbands:])
        subband_mean, modulator_mean = post_mean[:self.num_subbands], post_mean[self.num_subbands:]  # TODO: CHECK
        subband_cov = post_cov[:self.num_subbands, :self.num_subbands]
        modulator_cov = post_cov[self.num_subbands:, self.num_subbands:]
        sigma_points = cholesky(modulator_cov) @ x + modulator_mean
        modulator_mean_positive = self.weights @ self.link_fn(sigma_points)

        subband_var = np.diag(subband_cov)[..., None]
        mu = (modulator_mean_positive.T @ subband_mean)[:, 0]
        lognormpdf = -0.5 * np.log(2 * np.pi * self.variance) - 0.5 * (y - mu) ** 2 / self.variance
        const = -0.5 / self.variance * (modulator_mean_positive.T ** 2 @ subband_var)[:, 0]
        exp_log_lik = np.sum(w * (lognormpdf + const))
        return exp_log_lik

    def expected_log_likelihood_dm(self, y, post_mean, post_cov, cubature=None):
        """
        """
        dE_dm = grad(self.expected_log_likelihood, argnums=1)(y, post_mean, post_cov, cubature)
        return dE_dm

    def expected_log_likelihood_dm2(self, y, post_mean, post_cov, cubature=None):
        """
        """
        d2E_dm2 = jacrev(self.expected_log_likelihood_dm, argnums=1)(y, post_mean, post_cov, cubature)
        return np.squeeze(d2E_dm2)

    def variational_expectation(self, y, post_mean, post_cov, cubature=None):
        """
        Compute expected log likelihood via cubature:
        E[log p(yₙ|fₙ)] = ∫ log p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        """
        E = self.expected_log_likelihood(y, post_mean, post_cov, cubature)
        dE_dm = self.expected_log_likelihood_dm(y, post_mean, post_cov, cubature)
        d2E_dm2 = self.expected_log_likelihood_dm2(y, post_mean, post_cov, cubature)
        # d2E_dm2 = -ensure_positive_precision(-d2E_dm2)
        # return E, dE_dm, np.diag(np.diag(d2E_dm2))  # TODO: check this is the same as above
        return E, dE_dm, d2E_dm2

    def analytical_linearisation(self, m, sigma=None):
        """
        Compute the Jacobian of the state space observation model w.r.t. the
        function fₙ and the noise term σₙ.
        The implicit observation model is:
            h(fₙ,rₙ) = E[yₙ|fₙ] + √Cov[yₙ|fₙ] σₙ
        The Jacobians are evaluated at the means, fₙ=m, σₙ=0, to be used during
        Extended Kalman smoothing.
        """
        sigma = np.array([[0.0]]) if sigma is None else sigma

        Jf, Jsigma = jacrev(self.observation_model, argnums=(0, 1))(m, sigma)

        Hf = jacrev(self.jac_obs, argnums=0)(m, sigma)
        Hsigma = jacrev(self.jac_obs_sigma, argnums=1)(m, sigma)

        return Jf.T, np.swapaxes(Hf, axis1=0, axis2=2), Jsigma[None], Hsigma[None]


class NMF(NonnegativeMatrixFactorisation):
    pass


class AudioAmplitudeDemodulation(NonnegativeMatrixFactorisation):
    """
    The Audio Amplitude Demodulation likelihood
    """
    def __init__(self, num_components, variance=0.1):
        """
        param hyp: observation noise
        """
        self.transformed_variance = objax.TrainVar(np.array(softplus_inv(variance)))
        super().__init__(num_subbands=num_components,
                         num_modulators=num_components,
                         variance=variance)
        self.name = 'Audio Amplitude Demodulation'

    @property
    def weights(self):
        return np.eye(self.num_subbands)
