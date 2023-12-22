import objax
import jax.numpy as np
from jax import vmap
from .utils import diag, cho_factor, cho_solve, softplus, softplus_inv, transpose
from .basemodels import (
    GaussianProcess,
    SparseGaussianProcess,
    MarkovGaussianProcess,
    SparseMarkovGaussianProcess,
    MarkovMeanFieldGaussianProcess,
    SparseMarkovMeanFieldGaussianProcess,
    InfiniteHorizonGaussianProcess,
    SparseInfiniteHorizonGaussianProcess
)
from .inference import (
    Newton,
    VariationalInference,
    ExpectationPropagation,
    PosteriorLinearisation,
    PosteriorLinearisation2ndOrder,
    PosteriorLinearisation2ndOrderGaussNewton,
    PosteriorLinearisation2ndOrderQuasiNewton,
    Taylor,
    GaussNewton,
    VariationalGaussNewton,
    QuasiNewton,
    VariationalQuasiNewton,
    ExpectationPropagationQuasiNewton,
    # PosteriorLinearisationQuasiNewton,
    NewtonRiemann,
    VariationalInferenceRiemann,
    ExpectationPropagationRiemann,
    PosteriorLinearisation2ndOrderRiemann
)
from .kernels import Independent


# ############  Syntactic sugar adding the inference method functionality to the models  ################

# note: re-declaring the inputs here is not strictly necessary, but creates nice documentation


# ##### Variational Inference #####

class VariationalGP(VariationalInference, GaussianProcess):
    """
    Variational Gaussian process [1], adapted to use conjugate computation VI [2]
    :param kernel: a kernel object
    :param likelihood: a likelihood object
    :param X: inputs
    :param Y: observations

    [1] Opper, Archambeau: The Variational Gaussian Approximation Revisited, Neural Computation, 2009
    [2] Khan, Lin: Conugate-Computation Variational Inference - Converting Inference in Non-Conjugate Models in to
                   Inference in Conjugate Models, AISTATS 2017
    """
    def __init__(self, kernel, likelihood, X, Y):
        super().__init__(kernel, likelihood, X, Y)


class VariationalRiemannGP(VariationalInferenceRiemann, GaussianProcess):
    """
    Variational Gaussian process [1], adapted to use conjugate computation VI [2] with PSD guarantees [3].
    :param kernel: a kernel object
    :param likelihood: a likelihood object
    :param X: inputs
    :param Y: observations

    [1] Opper, Archambeau: The Variational Gaussian Approximation Revisited, Neural Computation, 2009
    [2] Khan, Lin: Conugate-Computation Variational Inference - Converting Inference in Non-Conjugate Models in to
                   Inference in Conjugate Models, AISTATS 2017
    [3] Lin, Schmidt & Khan: Handling the Positive-Definite Constraint in the Bayesian Learning Rule, ICML 2020
    """
    def __init__(self, kernel, likelihood, X, Y):
        super().__init__(kernel, likelihood, X, Y)


class SparseVariationalGP(VariationalInference, SparseGaussianProcess):
    """
    Sparse variational Gaussian process (SVGP) [1], adapted to use conjugate computation VI [2]
    :param kernel: a kernel object
    :param likelihood: a likelihood object
    :param X: inputs
    :param Y: observations
    :param Z: inducing inputs
    :param opt_z: boolean determining whether to optimise the inducing input locations

    [1] Hensman, Matthews, Ghahramani: Scalable Variational Gaussian Process Classification, AISTATS 2015
    [2] Khan, Lin: Conugate-Computation Variational Inference - Converting Inference in Non-Conjugate Models in to
                   Inference in Conjugate Models, AISTATS 2017
    """
    def __init__(self, kernel, likelihood, X, Y, Z, opt_z=False):
        super().__init__(kernel, likelihood, X, Y, Z, opt_z)


class SparseVariationalRiemannGP(VariationalInferenceRiemann, SparseGaussianProcess):
    """
    Sparse variational Gaussian process (SVGP) [1], adapted to use conjugate computation VI [2] with PSD guarantees [3].
    :param kernel: a kernel object
    :param likelihood: a likelihood object
    :param X: inputs
    :param Y: observations
    :param Z: inducing inputs
    :param opt_z: boolean determining whether to optimise the inducing input locations

    [1] Hensman, Matthews, Ghahramani: Scalable Variational Gaussian Process Classification, AISTATS 2015
    [2] Khan, Lin: Conugate-Computation Variational Inference - Converting Inference in Non-Conjugate Models in to
                   Inference in Conjugate Models, AISTATS 2017
    [3] Lin, Schmidt & Khan: Handling the Positive-Definite Constraint in the Bayesian Learning Rule, ICML 2020
    """
    def __init__(self, kernel, likelihood, X, Y, Z, opt_z=False):
        super().__init__(kernel, likelihood, X, Y, Z, opt_z)


SVGP = SparseVariationalGP


class MarkovVariationalGP(VariationalInference, MarkovGaussianProcess):
    """
    Markov variational Gaussian process: a VGP where the posterior is computed via
    (spatio-temporal) filtering and smoothing [1]
    :param kernel: a kernel object
    :param likelihood: a likelihood object
    :param X: inputs
    :param Y: observations
    :param R: spatial inputs
    :param parallel: boolean determining whether to run parallel filtering

    [1] Chang, Wilkinson, Khan, Solin: Fast Variational Learning in State Space Gaussian Process Models, MLSP 2020
    """
    def __init__(self, kernel, likelihood, X, Y, R=None, parallel=None):
        super().__init__(kernel, likelihood, X, Y, R=R, parallel=parallel)


class SparseMarkovVariationalGP(VariationalInference, SparseMarkovGaussianProcess):
    """
    Sparse Markov variational Gaussian process: a sparse VGP with inducing states, where the posterior is computed via
    (spatio-temporal) filtering and smoothing [1, 2].
    :param kernel: a kernel object
    :param likelihood: a likelihood object
    :param X: inputs
    :param Y: observations
    :param R: spatial inputs
    :param parallel: boolean determining whether to run parallel filtering
    :param Z: inducing inputs

    [1] Adam, Eleftheriadis, Durrande, Artemev, Hensman: Doubly Sparse Variational Gaussian Processes, AISTATS 2020
    [2] Wilkinson, Solin, Adam: Sparse Algorithms for Markovian Gaussian Processes, AISTATS 2021
    """
    def __init__(self, kernel, likelihood, X, Y, R=None, parallel=None, Z=None):
        super().__init__(kernel, likelihood, X, Y, R=R, parallel=parallel, Z=Z)


class MarkovVariationalMeanFieldGP(VariationalInference, MarkovMeanFieldGaussianProcess):
    pass


class SparseMarkovVariationalMeanFieldGP(VariationalInference, SparseMarkovMeanFieldGaussianProcess):
    pass


class InfiniteHorizonVariationalGP(VariationalInference, InfiniteHorizonGaussianProcess):
    """
    Infinite-horizon GP [1] with variational inference.
    :param kernel: a kernel object
    :param likelihood: a likelihood object
    :param X: inputs
    :param Y: observations
    :param R: spatial inputs
    :param dare_iters: number of iterations to run the DARE solver for

    [1] Solin, Hensman, Turner: Infinite-Horizon Gaussian Processes, NeurIPS 2018
    """
    def __init__(self, kernel, likelihood, X, Y, R=None, dare_iters=20, parallel=None):
        super().__init__(kernel, likelihood, X, Y, R=R, dare_iters=dare_iters, parallel=parallel)


class SparseInfiniteHorizonVariationalGP(VariationalInference, SparseInfiniteHorizonGaussianProcess):
    pass


# ##### Expectation Propagation #####

class ExpectationPropagationGP(ExpectationPropagation, GaussianProcess):
    """
    Expectation propagation Gaussian process (EPGP).
    :param kernel: a kernel object
    :param likelihood: a likelihood object
    :param X: inputs
    :param Y: observations
    [1] Minka: A Family of Algorithms for Approximate Bayesian Inference, Ph. D thesis 2000
    """
    def __init__(self, kernel, likelihood, X, Y, power=1.):
        self.power = power
        super().__init__(kernel, likelihood, X, Y)


class SparseExpectationPropagationGP(ExpectationPropagation, SparseGaussianProcess):
    """
    Sparse expectation propagation Gaussian process (SEPGP) [1].
    :param kernel: a kernel object
    :param likelihood: a likelihood object
    :param X: inputs
    :param Y: observations
    :param Z: inducing inputs
    :param opt_z: boolean determining whether to optimise the inducing input locations
    [1] Csato, Opper: Sparse on-line Gaussian processes, Neural Computation 2002
    [2] Bui, Yan, Turner: A Unifying Framework for Gaussian Process Pseudo Point Approximations Using
                          Power Expectation Propagation, JMLR 2017
    """
    def __init__(self, kernel, likelihood, X, Y, Z, power=1., opt_z=False):
        self.power = power
        super().__init__(kernel, likelihood, X, Y, Z, opt_z=opt_z)


class MarkovExpectationPropagationGP(ExpectationPropagation, MarkovGaussianProcess):
    """
    Markov EP Gaussian process: an EPGP where the posterior is computed via
    (spatio-temporal) filtering and smoothing [1].
    :param kernel: a kernel object
    :param likelihood: a likelihood object
    :param X: inputs
    :param Y: observations
    :param R: spatial inputs
    :param parallel: boolean determining whether to run parallel filtering

    [1] Wilkinson, Chang, Riis Andersen, Solin: State Space Expectation Propagation, ICML 2020
    """
    def __init__(self, kernel, likelihood, X, Y, R=None, power=1., parallel=None):
        self.power = power
        super().__init__(kernel, likelihood, X, Y, R=R, parallel=parallel)


class SparseMarkovExpectationPropagationGP(ExpectationPropagation, SparseMarkovGaussianProcess):
    """
    Sparse Markov EP Gaussian process: a sparse EPGP with inducing states, where the posterior is computed via
    (spatio-temporal) filtering and smoothing [1].
    :param kernel: a kernel object
    :param likelihood: a likelihood object
    :param X: inputs
    :param Y: observations
    :param R: spatial inputs
    :param parallel: boolean determining whether to run parallel filtering
    :param Z: inducing inputs

    [1] Wilkinson, Solin, Adam: Sparse Algorithms for Markovian Gaussian Processes, AISTATS 2021
    """
    def __init__(self, kernel, likelihood, X, Y, R=None, power=1., parallel=None, Z=None):
        self.power = power
        super().__init__(kernel, likelihood, X, Y, R=R, parallel=parallel, Z=Z)


class MarkovExpectationPropagationMeanFieldGP(ExpectationPropagation, MarkovMeanFieldGaussianProcess):
    pass


class SparseMarkovExpectationPropagationMeanFieldGP(ExpectationPropagation, SparseMarkovMeanFieldGaussianProcess):
    pass


class InfiniteHorizonExpectationPropagationGP(ExpectationPropagation, InfiniteHorizonGaussianProcess):
    """
    Infinite-horizon GP [1] with expectation propagation.
    :param kernel: a kernel object
    :param likelihood: a likelihood object
    :param X: inputs
    :param Y: observations
    :param R: spatial inputs
    :param dare_iters: number of iterations to run the DARE solver for

    [1] Solin, Hensman, Turner: Infinite-Horizon Gaussian Processes, NeurIPS 2018
    """
    def __init__(self, kernel, likelihood, X, Y, R=None, power=1., dare_iters=20, parallel=None):
        self.power = power
        super().__init__(kernel, likelihood, X, Y, R=R, dare_iters=dare_iters, parallel=parallel)


class SparseInfiniteHorizonExpectationPropagationGP(ExpectationPropagation, SparseInfiniteHorizonGaussianProcess):
    pass


# ##### Newton / Laplace #####

class NewtonGP(Newton, GaussianProcess):
    """
    [1] Rasmussen, Williams: Gaussian Processes for Machine Learning, 2006
    """
    def __init__(self, kernel, likelihood, X, Y):
        super().__init__(kernel, likelihood, X, Y)


class SparseNewtonGP(Newton, SparseGaussianProcess):
    def __init__(self, kernel, likelihood, X, Y, Z, opt_z=False):
        super().__init__(kernel, likelihood, X, Y, Z, opt_z=opt_z)


class MarkovNewtonGP(Newton, MarkovGaussianProcess):
    def __init__(self, kernel, likelihood, X, Y, R=None, parallel=None):
        super().__init__(kernel, likelihood, X, Y, R=R, parallel=parallel)


class SparseMarkovNewtonGP(Newton, SparseMarkovGaussianProcess):
    def __init__(self, kernel, likelihood, X, Y, R=None, parallel=None, Z=None):
        super().__init__(kernel, likelihood, X, Y, R=R, parallel=parallel, Z=Z)


class MarkovNewtonMeanFieldGP(Newton, MarkovMeanFieldGaussianProcess):
    pass


class SparseMarkovNewtonMeanFieldGP(Newton, SparseMarkovMeanFieldGaussianProcess):
    pass


class InfiniteHorizonNewtonGP(Newton, InfiniteHorizonGaussianProcess):
    def __init__(self, kernel, likelihood, X, Y, R=None, dare_iters=20, parallel=None):
        super().__init__(kernel, likelihood, X, Y, R=R, dare_iters=dare_iters, parallel=parallel)


class SparseInfiniteHorizonNewtonGP(Newton, SparseInfiniteHorizonGaussianProcess):
    pass


LaplaceGP = NewtonGP

SparseLaplaceGP = SparseNewtonGP

MarkovLaplaceGP = MarkovNewtonGP

SparseMarkovLaplaceGP = SparseMarkovNewtonGP

MarkovLaplaceGPMeanField = MarkovNewtonMeanFieldGP

SparseMarkovLaplaceGPMeanField = SparseMarkovNewtonMeanFieldGP

InfiniteHorizonLaplaceGP = InfiniteHorizonNewtonGP

SparseInfiniteHorizonLaplaceGP = SparseInfiniteHorizonNewtonGP


# ##### Posterior Linearisation #####

class PosteriorLinearisationGP(PosteriorLinearisation, GaussianProcess):
    """
    [1] Garcia-Fernandez, Tronarp, Sarkka: Gaussian Process Classification
                                           Using Posterior Linearization, IEEE Signal Processing 2019
    """
    def __init__(self, kernel, likelihood, X, Y):
        super().__init__(kernel, likelihood, X, Y)


# class PosteriorLinearisationNewtonGP(PosteriorLinearisationNewton, GaussianProcess):
#     pass


class SparsePosteriorLinearisationGP(PosteriorLinearisation, SparseGaussianProcess):
    def __init__(self, kernel, likelihood, X, Y, Z, opt_z=False):
        super().__init__(kernel, likelihood, X, Y, Z, opt_z=opt_z)


class MarkovPosteriorLinearisationGP(PosteriorLinearisation, MarkovGaussianProcess):
    """
    [1] Garcia-Fernandez, Svensson, Sarkka: Iterated Posterior Linearization Smoother, IEEE Automatic Control 2016
    [2] Wilkinson, Chang, Riis Andersen, Solin: State Space Expectation Propagation, ICML 2020
    """

    def __init__(self, kernel, likelihood, X, Y, R=None, parallel=None):
        super().__init__(kernel, likelihood, X, Y, R=R, parallel=parallel)


class SparseMarkovPosteriorLinearisationGP(PosteriorLinearisation, SparseMarkovGaussianProcess):
    """
    [1] Wilkinson, Solin, Adam: Sparse Algorithms for Markovian Gaussian Processes, AISTATS 2021
    """

    def __init__(self, kernel, likelihood, X, Y, R=None, parallel=None, Z=None):
        super().__init__(kernel, likelihood, X, Y, R=R, parallel=parallel, Z=Z)


class MarkovPosteriorLinearisationMeanFieldGP(PosteriorLinearisation, MarkovMeanFieldGaussianProcess):
    pass


class SparseMarkovPosteriorLinearisationMeanFieldGP(PosteriorLinearisation, SparseMarkovMeanFieldGaussianProcess):
    pass


class InfiniteHorizonPosteriorLinearisationGP(PosteriorLinearisation, InfiniteHorizonGaussianProcess):
    def __init__(self, kernel, likelihood, X, Y, R=None, dare_iters=20, parallel=None):
        super().__init__(kernel, likelihood, X, Y, R=R, dare_iters=dare_iters, parallel=parallel)


class SparseInfiniteHorizonPosteriorLinearisationGP(PosteriorLinearisation, SparseInfiniteHorizonGaussianProcess):
    pass


# ##### Taylor #####

class TaylorGP(Taylor, GaussianProcess):
    """
    [1] Steinberg, Bonilla: Extended and Unscented Gaussian Processes, NeurIPS 2014
    """
    def __init__(self, kernel, likelihood, X, Y):
        super().__init__(kernel, likelihood, X, Y)


class SparseTaylorGP(Taylor, SparseGaussianProcess):
    def __init__(self, kernel, likelihood, X, Y, Z, opt_z=False):
        super().__init__(kernel, likelihood, X, Y, Z, opt_z=opt_z)


class MarkovTaylorGP(Taylor, MarkovGaussianProcess):
    """
    [1] Bell: The Iterated Kalman Smoother as a Gauss-Newton method, SIAM Journal on Optimization 1994
    """
    def __init__(self, kernel, likelihood, X, Y, R=None, parallel=None):
        super().__init__(kernel, likelihood, X, Y, R=R, parallel=parallel)


class SparseMarkovTaylorGP(Taylor, SparseMarkovGaussianProcess):
    def __init__(self, kernel, likelihood, X, Y, R=None, parallel=None, Z=None):
        super().__init__(kernel, likelihood, X, Y, R=R, parallel=parallel, Z=Z)


class MarkovTaylorMeanFieldGP(Taylor, MarkovMeanFieldGaussianProcess):
    pass


class SparseMarkovTaylorMeanFieldGP(Taylor, SparseMarkovMeanFieldGaussianProcess):
    pass


class InfiniteHorizonTaylorGP(Taylor, InfiniteHorizonGaussianProcess):
    def __init__(self, kernel, likelihood, X, Y, R=None, dare_iters=20, parallel=None):
        super().__init__(kernel, likelihood, X, Y, R=R, dare_iters=dare_iters, parallel=parallel)


class SparseInfiniteHorizonTaylorGP(Taylor, SparseInfiniteHorizonGaussianProcess):
    pass


# Extensions to posterior linearisation

class MarkovPosteriorLinearisation2ndOrderGP(PosteriorLinearisation2ndOrder, MarkovGaussianProcess):
    pass


class MarkovPosteriorLinearisation2ndOrderGaussNewtonGP(PosteriorLinearisation2ndOrderGaussNewton,
                                                        MarkovGaussianProcess):
    pass


class MarkovPosteriorLinearisation2ndOrderRiemannGP(PosteriorLinearisation2ndOrderRiemann, MarkovGaussianProcess):
    pass


# Gauss-Newton approximations

class GaussNewtonGP(GaussNewton, GaussianProcess):
    def __init__(self, kernel, likelihood, X, Y):
        super().__init__(kernel, likelihood, X, Y)


LaplaceGaussNewtonGP = GaussNewtonGP


class MarkovGaussNewtonGP(GaussNewton, MarkovGaussianProcess):
    def __init__(self, kernel, likelihood, X, Y, R=None, parallel=None):
        super().__init__(kernel, likelihood, X, Y, R=R, parallel=parallel)


MarkovLaplaceGaussNewtonGP = MarkovGaussNewtonGP


class SparseGaussNewtonGP(GaussNewton, SparseGaussianProcess):
    def __init__(self, kernel, likelihood, X, Y, Z, opt_z=False):
        super().__init__(kernel, likelihood, X, Y, Z, opt_z)


SparseLaplaceGaussNewtonGP = SparseGaussNewtonGP


class VariationalGaussNewtonGP(VariationalGaussNewton, GaussianProcess):
    def __init__(self, kernel, likelihood, X, Y):
        super().__init__(kernel, likelihood, X, Y)


class MarkovVariationalGaussNewtonGP(VariationalGaussNewton, MarkovGaussianProcess):
    def __init__(self, kernel, likelihood, X, Y, R=None, parallel=None):
        super().__init__(kernel, likelihood, X, Y, R=R, parallel=parallel)


class SparseVariationalGaussNewtonGP(VariationalGaussNewton, SparseGaussianProcess):
    def __init__(self, kernel, likelihood, X, Y, Z, opt_z=False):
        super().__init__(kernel, likelihood, X, Y, Z, opt_z)


# Quasi-Newton approximations

# --- quasi-Newton ---

class QuasiNewtonGP(QuasiNewton, GaussianProcess):
    def __init__(self, kernel, likelihood, X, Y):
        super().__init__(kernel, likelihood, X, Y)
        self.mean_prev = objax.StateVar(self.pseudo_likelihood.mean)
        self.jacobian_prev = objax.StateVar(np.zeros([self.num_data, self.func_dim, 1]))
        self.hessian_approx = objax.StateVar(-1e2 * np.tile(np.eye(self.func_dim), [self.num_data, 1, 1]))


LaplaceQuasiNewtonGP = QuasiNewtonGP


class VariationalQuasiNewtonGP(VariationalQuasiNewton, GaussianProcess):
    def __init__(self, kernel, likelihood, X, Y, fullcov=True):
        self.fullcov = fullcov
        super().__init__(kernel, likelihood, X, Y)
        if fullcov:
            self.mean_prev = objax.StateVar(
                np.concatenate([self.pseudo_likelihood.mean,
                                np.reshape(self.pseudo_likelihood.covariance, (self.num_data, -1, 1))],
                               axis=1)
            )
            dim = self.mean_prev.value.shape[1]
            self.jacobian_prev = objax.StateVar(np.zeros([self.num_data, dim, 1]))
            self.hessian_approx = objax.StateVar(-1e2 * np.tile(np.eye(dim), [self.num_data, 1, 1]))
        else:
            self.mean_prev = objax.StateVar(
                np.concatenate([self.pseudo_likelihood.mean,
                                diag(self.pseudo_likelihood.covariance)[..., None]], axis=1)
            )
            self.jacobian_prev = objax.StateVar(np.zeros([self.num_data, 2 * self.func_dim, 1]))
            self.hessian_approx = objax.StateVar(-1e2 * np.tile(np.eye(2 * self.func_dim), [self.num_data, 1, 1]))


class ExpectationPropagationQuasiNewtonGP(ExpectationPropagationQuasiNewton, GaussianProcess):
    def __init__(self, kernel, likelihood, X, Y, power=1., fullcov=True):
        self.power = power
        self.fullcov = fullcov
        super().__init__(kernel, likelihood, X, Y)
        if fullcov:
            self.mean_prev = objax.StateVar(
                np.concatenate([self.pseudo_likelihood.mean,
                                np.reshape(self.pseudo_likelihood.covariance, (self.num_data, -1, 1))],
                               axis=1)
            )
            dim = self.mean_prev.value.shape[1]
            self.jacobian_prev = objax.StateVar(np.zeros([self.num_data, dim, 1]))
            self.hessian_approx = objax.StateVar(-1e2 * np.tile(np.eye(dim), [self.num_data, 1, 1]))
        else:
            self.mean_prev = objax.StateVar(
                np.concatenate([self.pseudo_likelihood.mean,
                                diag(self.pseudo_likelihood.covariance)[..., None]], axis=1)
            )
            self.jacobian_prev = objax.StateVar(np.zeros([self.num_data, 2 * self.func_dim, 1]))
            self.hessian_approx = objax.StateVar(-1e2 * np.tile(np.eye(2 * self.func_dim), [self.num_data, 1, 1]))


# class PosteriorLinearisationQuasiNewtonGP(PosteriorLinearisationQuasiNewton, GaussianProcess):
#     def __init__(self, kernel, likelihood, X, Y, fullcov=True):
#         self.fullcov = fullcov
#         super().__init__(kernel, likelihood, X, Y)
#         if fullcov:
#             self.mean_prev = objax.StateVar(
#                 np.concatenate([self.pseudo_likelihood.mean,
#                                 np.reshape(self.pseudo_likelihood.covariance, (self.num_data, -1, 1))],
#                                axis=1)
#             )
#             dim = self.mean_prev.value.shape[1]
#             self.jacobian_prev = objax.StateVar(np.zeros([self.num_data, dim, 1]))
#             self.hessian_approx = objax.StateVar(-1e2 * np.tile(np.eye(dim), [self.num_data, 1, 1]))
#         else:
#             self.mean_prev = objax.StateVar(
#                 np.concatenate([self.pseudo_likelihood.mean,
#                                 diag(self.pseudo_likelihood.covariance)[..., None]], axis=1)
#             )
#             self.jacobian_prev = objax.StateVar(np.zeros([self.num_data, 2 * self.func_dim, 1]))
#             self.hessian_approx = objax.StateVar(-1e2 * np.tile(np.eye(2 * self.func_dim), [self.num_data, 1, 1]))


class PosteriorLinearisation2ndOrderQuasiNewtonGP(PosteriorLinearisation2ndOrderQuasiNewton, GaussianProcess):
    def __init__(self, kernel, likelihood, X, Y, fullcov=True):
        self.fullcov = fullcov
        super().__init__(kernel, likelihood, X, Y)
        if fullcov:
            self.mean_prev = objax.StateVar(
                np.concatenate([self.pseudo_likelihood.mean,
                                np.reshape(self.pseudo_likelihood.covariance, (self.num_data, -1, 1))],
                               axis=1)
            )
            dim = self.mean_prev.value.shape[1]
            self.jacobian_prev = objax.StateVar(np.zeros([self.num_data, dim, 1]))
            self.hessian_approx = objax.StateVar(-1e2 * np.tile(np.eye(dim), [self.num_data, 1, 1]))
        else:
            self.mean_prev = objax.StateVar(
                np.concatenate([self.pseudo_likelihood.mean,
                                diag(self.pseudo_likelihood.covariance)[..., None]], axis=1)
            )
            self.jacobian_prev = objax.StateVar(np.zeros([self.num_data, 2 * self.func_dim, 1]))
            self.hessian_approx = objax.StateVar(-1e2 * np.tile(np.eye(2 * self.func_dim), [self.num_data, 1, 1]))


# --- Sparse Quasi-Newton ---

class SparseQuasiNewtonGP(QuasiNewton, SparseGaussianProcess):
    def __init__(self, kernel, likelihood, X, Y, Z, opt_z=False):
        super().__init__(kernel, likelihood, X, Y, Z, opt_z=opt_z)
        self.mean_prev = objax.StateVar(self.pseudo_likelihood.mean)
        self.jacobian_prev = objax.StateVar(np.zeros([self.num_data, self.func_dim, 1]))
        self.hessian_approx = objax.StateVar(-1e2 * np.tile(np.eye(self.func_dim), [self.num_data, 1, 1]))


SparseLaplaceQuasiNewtonGP = SparseQuasiNewtonGP


class SparseVariationalQuasiNewtonGP(VariationalQuasiNewton, SparseGaussianProcess):
    def __init__(self, kernel, likelihood, X, Y, Z, opt_z=False, fullcov=True):
        self.fullcov = fullcov
        super().__init__(kernel, likelihood, X, Y, Z, opt_z=opt_z)
        if fullcov:
            self.mean_prev = objax.StateVar(
                np.concatenate([self.pseudo_likelihood.mean,
                                np.reshape(self.pseudo_likelihood.covariance, (self.num_data, -1, 1))],
                               axis=1)
            )
            dim = self.mean_prev.value.shape[1]
            self.jacobian_prev = objax.StateVar(np.zeros([self.num_data, dim, 1]))
            self.hessian_approx = objax.StateVar(-1e2 * np.tile(np.eye(dim), [self.num_data, 1, 1]))
        else:
            self.mean_prev = objax.StateVar(
                np.concatenate([self.pseudo_likelihood.mean,
                                diag(self.pseudo_likelihood.covariance)[..., None]], axis=1)
            )
            self.jacobian_prev = objax.StateVar(np.zeros([self.num_data, 2 * self.func_dim, 1]))
            self.hessian_approx = objax.StateVar(-1e2 * np.tile(np.eye(2 * self.func_dim), [self.num_data, 1, 1]))


class SparseExpectationPropagationQuasiNewtonGP(ExpectationPropagationQuasiNewton, SparseGaussianProcess):
    def __init__(self, kernel, likelihood, X, Y, Z, power=1., opt_z=False, fullcov=True):
        self.power = power
        self.fullcov = fullcov
        super().__init__(kernel, likelihood, X, Y, Z, opt_z=opt_z)
        if fullcov:
            self.mean_prev = objax.StateVar(
                np.concatenate([self.pseudo_likelihood.mean,
                                np.reshape(self.pseudo_likelihood.covariance, (self.num_data, -1, 1))],
                               axis=1)
            )
            dim = self.mean_prev.value.shape[1]
            self.jacobian_prev = objax.StateVar(np.zeros([self.num_data, dim, 1]))
            self.hessian_approx = objax.StateVar(-1e2 * np.tile(np.eye(dim), [self.num_data, 1, 1]))
        else:
            self.mean_prev = objax.StateVar(
                np.concatenate([self.pseudo_likelihood.mean,
                                diag(self.pseudo_likelihood.covariance)[..., None]], axis=1)
            )
            self.jacobian_prev = objax.StateVar(np.zeros([self.num_data, 2 * self.func_dim, 1]))
            self.hessian_approx = objax.StateVar(-1e2 * np.tile(np.eye(2 * self.func_dim), [self.num_data, 1, 1]))


# class SparsePosteriorLinearisationQuasiNewtonGP(PosteriorLinearisationQuasiNewton, SparseGaussianProcess):
#     def __init__(self, kernel, likelihood, X, Y, Z, opt_z=False, fullcov=True):
#         self.fullcov = fullcov
#         super().__init__(kernel, likelihood, X, Y, Z, opt_z=opt_z)
#         if fullcov:
#             self.mean_prev = objax.StateVar(
#                 np.concatenate([self.pseudo_likelihood.mean,
#                                 np.reshape(self.pseudo_likelihood.covariance, (self.num_data, -1, 1))],
#                                axis=1)
#             )
#             dim = self.mean_prev.value.shape[1]
#             self.jacobian_prev = objax.StateVar(np.zeros([self.num_data, dim, 1]))
#             self.hessian_approx = objax.StateVar(-1e2 * np.tile(np.eye(dim), [self.num_data, 1, 1]))
#         else:
#             self.mean_prev = objax.StateVar(
#                 np.concatenate([self.pseudo_likelihood.mean,
#                                 diag(self.pseudo_likelihood.covariance)[..., None]], axis=1)
#             )
#             self.jacobian_prev = objax.StateVar(np.zeros([self.num_data, 2 * self.func_dim, 1]))
#             self.hessian_approx = objax.StateVar(-1e2 * np.tile(np.eye(2 * self.func_dim), [self.num_data, 1, 1]))


class SparsePosteriorLinearisation2ndOrderQuasiNewtonGP(PosteriorLinearisation2ndOrderQuasiNewton,
                                                        SparseGaussianProcess):
    def __init__(self, kernel, likelihood, X, Y, Z, opt_z=False, fullcov=True):
        self.fullcov = fullcov
        super().__init__(kernel, likelihood, X, Y, Z, opt_z=opt_z)
        if fullcov:
            self.mean_prev = objax.StateVar(
                np.concatenate([self.pseudo_likelihood.mean,
                                np.reshape(self.pseudo_likelihood.covariance, (self.num_data, -1, 1))],
                               axis=1)
            )
            dim = self.mean_prev.value.shape[1]
            self.jacobian_prev = objax.StateVar(np.zeros([self.num_data, dim, 1]))
            self.hessian_approx = objax.StateVar(-1e2 * np.tile(np.eye(dim), [self.num_data, 1, 1]))
        else:
            self.mean_prev = objax.StateVar(
                np.concatenate([self.pseudo_likelihood.mean,
                                diag(self.pseudo_likelihood.covariance)[..., None]], axis=1)
            )
            self.jacobian_prev = objax.StateVar(np.zeros([self.num_data, 2 * self.func_dim, 1]))
            self.hessian_approx = objax.StateVar(-1e2 * np.tile(np.eye(2 * self.func_dim), [self.num_data, 1, 1]))


# --- Markov quasi-Newton ---


class MarkovQuasiNewtonGP(QuasiNewton, MarkovGaussianProcess):
    def __init__(self, kernel, likelihood, X, Y, R=None, parallel=None):
        super().__init__(kernel, likelihood, X, Y, R=R, parallel=parallel)
        self.mean_prev = objax.StateVar(self.pseudo_likelihood.mean)
        self.jacobian_prev = objax.StateVar(np.zeros([self.num_data, self.func_dim, 1]))
        self.hessian_approx = objax.StateVar(-1e2 * np.tile(np.eye(self.func_dim), [self.num_data, 1, 1]))


MarkovLaplaceQuasiNewtonGP = MarkovQuasiNewtonGP


class MarkovVariationalQuasiNewtonGP(VariationalQuasiNewton, MarkovGaussianProcess):
    def __init__(self, kernel, likelihood, X, Y, R=None, parallel=None, fullcov=True):
        self.fullcov = fullcov
        super().__init__(kernel, likelihood, X, Y, R=R, parallel=parallel)
        if fullcov:
            self.mean_prev = objax.StateVar(
                np.concatenate([self.pseudo_likelihood.mean,
                                np.reshape(self.pseudo_likelihood.covariance, (self.num_data, -1, 1))],
                               axis=1)
            )
            dim = self.mean_prev.value.shape[1]
            self.jacobian_prev = objax.StateVar(np.zeros([self.num_data, dim, 1]))
            self.hessian_approx = objax.StateVar(-1e2 * np.tile(np.eye(dim), [self.num_data, 1, 1]))
        else:
            self.mean_prev = objax.StateVar(
                np.concatenate([self.pseudo_likelihood.mean,
                                diag(self.pseudo_likelihood.covariance)[..., None]], axis=1)
            )
            self.jacobian_prev = objax.StateVar(np.zeros([self.num_data, 2 * self.func_dim, 1]))
            self.hessian_approx = objax.StateVar(-1e2 * np.tile(np.eye(2 * self.func_dim), [self.num_data, 1, 1]))


class MarkovExpectationPropagationQuasiNewtonGP(ExpectationPropagationQuasiNewton, MarkovGaussianProcess):
    def __init__(self, kernel, likelihood, X, Y, R=None, power=1., parallel=None, fullcov=True):
        self.power = power
        self.fullcov = fullcov
        super().__init__(kernel, likelihood, X, Y, R=R, parallel=parallel)
        if fullcov:
            self.mean_prev = objax.StateVar(
                np.concatenate([self.pseudo_likelihood.mean,
                                np.reshape(self.pseudo_likelihood.covariance, (self.num_data, -1, 1))],
                               axis=1)
            )
            dim = self.mean_prev.value.shape[1]
            self.jacobian_prev = objax.StateVar(np.zeros([self.num_data, dim, 1]))
            self.hessian_approx = objax.StateVar(-1e2 * np.tile(np.eye(dim), [self.num_data, 1, 1]))
        else:
            self.mean_prev = objax.StateVar(
                np.concatenate([self.pseudo_likelihood.mean,
                                diag(self.pseudo_likelihood.covariance)[..., None]], axis=1)
            )
            self.jacobian_prev = objax.StateVar(np.zeros([self.num_data, 2 * self.func_dim, 1]))
            self.hessian_approx = objax.StateVar(-1e2 * np.tile(np.eye(2 * self.func_dim), [self.num_data, 1, 1]))


# class MarkovPosteriorLinearisationQuasiNewtonGP(PosteriorLinearisationQuasiNewton, MarkovGaussianProcess):
#     def __init__(self, kernel, likelihood, X, Y, R=None, parallel=None, fullcov=True):
#         self.fullcov = fullcov
#         super().__init__(kernel, likelihood, X, Y, R=R, parallel=parallel)
#         if fullcov:
#             self.mean_prev = objax.StateVar(
#                 np.concatenate([self.pseudo_likelihood.mean,
#                                 np.reshape(self.pseudo_likelihood.covariance, (self.num_data, -1, 1))],
#                                axis=1)
#             )
#             dim = self.mean_prev.value.shape[1]
#             self.jacobian_prev = objax.StateVar(np.zeros([self.num_data, dim, 1]))
#             self.hessian_approx = objax.StateVar(-1e2 * np.tile(np.eye(dim), [self.num_data, 1, 1]))
#         else:
#             self.mean_prev = objax.StateVar(
#                 np.concatenate([self.pseudo_likelihood.mean,
#                                 diag(self.pseudo_likelihood.covariance)[..., None]], axis=1)
#             )
#             self.jacobian_prev = objax.StateVar(np.zeros([self.num_data, 2 * self.func_dim, 1]))
#             self.hessian_approx = objax.StateVar(-1e2 * np.tile(np.eye(2 * self.func_dim), [self.num_data, 1, 1]))


class MarkovPosteriorLinearisation2ndOrderQuasiNewtonGP(PosteriorLinearisation2ndOrderQuasiNewton,
                                                        MarkovGaussianProcess):
    def __init__(self, kernel, likelihood, X, Y, R=None, parallel=None, fullcov=True):
        self.fullcov = fullcov
        super().__init__(kernel, likelihood, X, Y, R=R, parallel=parallel)
        if fullcov:
            self.mean_prev = objax.StateVar(
                np.concatenate([self.pseudo_likelihood.mean,
                                np.reshape(self.pseudo_likelihood.covariance, (self.num_data, -1, 1))],
                               axis=1)
            )
            dim = self.mean_prev.value.shape[1]
            self.jacobian_prev = objax.StateVar(np.zeros([self.num_data, dim, 1]))
            self.hessian_approx = objax.StateVar(-1e2 * np.tile(np.eye(dim), [self.num_data, 1, 1]))
        else:
            self.mean_prev = objax.StateVar(
                np.concatenate([self.pseudo_likelihood.mean,
                                diag(self.pseudo_likelihood.covariance)[..., None]], axis=1)
            )
            self.jacobian_prev = objax.StateVar(np.zeros([self.num_data, 2 * self.func_dim, 1]))
            self.hessian_approx = objax.StateVar(-1e2 * np.tile(np.eye(2 * self.func_dim), [self.num_data, 1, 1]))


# PSD constraints via Riemannian gradients

class MarkovVariationalRiemannGP(VariationalInferenceRiemann, MarkovGaussianProcess):
    """
    Markov variational Gaussian process: a VGP where the posterior is computed via
    (spatio-temporal) filtering and smoothing [1] with PSD constraints via Riemannian gradients [2].
    :param kernel: a kernel object
    :param likelihood: a likelihood object
    :param X: inputs
    :param Y: observations
    :param R: spatial inputs
    :param parallel: boolean determining whether to run parallel filtering

    [1] Chang, Wilkinson, Khan, Solin: Fast Variational Learning in State Space Gaussian Process Models, MLSP 2020
    [2] Lin, Schmidt, Khan: Handling the Positive-Definite Constraint in the Bayesian Learning Rule, ICML 2020
    """
    def __init__(self, kernel, likelihood, X, Y, R=None, parallel=None):
        super().__init__(kernel, likelihood, X, Y, R=R, parallel=parallel)


class MarkovExpectationPropagationRiemannGP(ExpectationPropagationRiemann, MarkovGaussianProcess):
    """
    Markov EP Gaussian process: an EPGP where the posterior is computed via
    (spatio-temporal) filtering and smoothing [1] with PSD constraints via Riemannian gradients [2].
    :param kernel: a kernel object
    :param likelihood: a likelihood object
    :param X: inputs
    :param Y: observations
    :param R: spatial inputs
    :param parallel: boolean determining whether to run parallel filtering

    [1] Wilkinson, Chang, Riis Andersen, Solin: State Space Expectation Propagation, ICML 2020
    [2] Lin, Schmidt, Khan: Handling the Positive-Definite Constraint in the Bayesian Learning Rule, ICML 2020
    """
    def __init__(self, kernel, likelihood, X, Y, R=None, power=1., parallel=None):
        self.power = power
        super().__init__(kernel, likelihood, X, Y, R=R, parallel=parallel)


class MarkovNewtonRiemannGP(NewtonRiemann, MarkovGaussianProcess):
    """
    Markov Laplace Gaussian process with PSD constraints via Riemannian gradients [1].
    :param kernel: a kernel object
    :param likelihood: a likelihood object
    :param X: inputs
    :param Y: observations
    :param R: spatial inputs
    :param parallel: boolean determining whether to run parallel filtering

    [1] Lin, Schmidt, Khan: Handling the Positive-Definite Constraint in the Bayesian Learning Rule, ICML 2020
    """
    def __init__(self, kernel, likelihood, X, Y, R=None, parallel=None):
        super().__init__(kernel, likelihood, X, Y, R=R, parallel=parallel)


MarkovLaplaceRiemannGP = MarkovNewtonRiemannGP


class TrainableDiagonalGaussianDistribution(objax.Module):

    def __init__(self, mean, variance):
        self.mean_ = objax.TrainVar(mean)
        self.transformed_variance = objax.TrainVar(vmap(softplus_inv)(variance))

    def __call__(self):
        return self.mean, self.covariance

    @property
    def mean(self):
        return self.mean_.value

    @property
    def variance(self):
        return softplus(self.transformed_variance.value)

    @property
    def covariance(self):
        return vmap(np.diag)(self.variance)

    @property
    def nat1(self):
        chol = cho_factor(self.covariance, lower=True)
        return cho_solve(chol, self.mean)

    @property
    def nat2(self):
        chol = cho_factor(self.covariance, lower=True)
        return cho_solve(chol, np.tile(np.eye(self.covariance.shape[1]), [self.covariance.shape[0], 1, 1]))


class TrainableGaussianDistribution(objax.Module):

    def __init__(self, mean, covariance):
        self.dim = mean.shape[1]
        cholcov, _ = cho_factor(covariance, lower=True)
        self.mean_ = objax.TrainVar(mean)
        self.transformed_covariance = objax.TrainVar(vmap(self.get_tril, [0, None])(cholcov, self.dim))

    def __call__(self):
        return self.mean, self.covariance

    @staticmethod
    def get_tril(chol, dim):
        return chol[np.tril_indices(dim)]

    def fill_lower_tri(self, v):
        idx = np.tril_indices(self.dim)
        return np.zeros((self.dim, self.dim), dtype=v.dtype).at[idx].set(v)

    @property
    def mean(self):
        return self.mean_.value

    @property
    def covariance(self):
        chol_low = vmap(self.fill_lower_tri)(self.transformed_covariance.value)
        return transpose(chol_low) @ chol_low

    @property
    def nat1(self):
        chol = cho_factor(self.covariance, lower=True)
        return cho_solve(chol, self.mean)

    @property
    def nat2(self):
        chol = cho_factor(self.covariance, lower=True)
        return cho_solve(chol, np.tile(np.eye(self.covariance.shape[1]), [self.covariance.shape[0], 1, 1]))


class FirstOrderVariationalGP(VariationalGP):

    def __init__(self, kernel, likelihood, X, Y):
        super().__init__(kernel, likelihood, X, Y)
        if isinstance(self.kernel, Independent):
            pseudo_lik_size = self.func_dim  # the multi-latent case
        else:
            pseudo_lik_size = self.obs_dim
        # self.pseudo_likelihood = TrainableDiagonalGaussianDistribution(
        #     mean=np.zeros([self.num_data, pseudo_lik_size, 1]),
        #     variance=1e2 * np.ones([self.num_data, pseudo_lik_size])
        # )
        self.pseudo_likelihood = TrainableGaussianDistribution(
            mean=np.zeros([self.num_data, pseudo_lik_size, 1]),
            covariance=1e2 * np.tile(np.eye(pseudo_lik_size), [self.num_data, 1, 1])
        )

    def energy(self, **kwargs):
        """
        """
        self.update_posterior()
        return super().energy(**kwargs)


class FirstOrderMarkovVariationalGP(MarkovVariationalGP):

    def __init__(self, kernel, likelihood, X, Y):
        super().__init__(kernel, likelihood, X, Y)
        if isinstance(self.kernel, Independent):
            pseudo_lik_size = self.func_dim  # the multi-latent case
        else:
            pseudo_lik_size = self.obs_dim
        self.pseudo_likelihood = TrainableGaussianDistribution(
            mean=np.zeros([self.num_data, pseudo_lik_size, 1]),
            covariance=1e2 * np.tile(np.eye(pseudo_lik_size), [self.num_data, 1, 1])
        )

    def energy(self, **kwargs):
        """
        """
        self.update_posterior()
        return super().energy(**kwargs)
