from .basemodels import (
    GP,
    SparseGP,
    MarkovGP,
    SparseMarkovGP,
)
from .inference import (
    VariationalInference,
    ExpectationPropagation,
    Laplace,
    PosteriorLinearisation,
    Taylor
)


# ############  Syntactic sugar adding the inference method functionality to the models  ################

# note: re-declaring the inputs here is not strictly necessary, but creates nice documentation


# ##### Variational Inference #####

class VariationalGP(VariationalInference, GP):
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


class SparseVariationalGP(VariationalInference, SparseGP):
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


class MarkovVariationalGP(VariationalInference, MarkovGP):
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


class SparseMarkovVariationalGP(VariationalInference, SparseMarkovGP):
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


# ##### Expectation Propagation #####

class ExpectationPropagationGP(ExpectationPropagation, GP):
    """
    Expectation propagation Gaussian process (EPGP).
    :param kernel: a kernel object
    :param likelihood: a likelihood object
    :param X: inputs
    :param Y: observations
    [1] Minka: Expectation Propagation for Approximate Bayesian Inference, Ph. D thesis 2000
    """
    def __init__(self, kernel, likelihood, X, Y):
        super().__init__(kernel, likelihood, X, Y)


class SparseExpectationPropagationGP(ExpectationPropagation, SparseGP):
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
    def __init__(self, kernel, likelihood, X, Y, Z, opt_z=False):
        super().__init__(kernel, likelihood, X, Y, Z, opt_z=opt_z)


class MarkovExpectationPropagationGP(ExpectationPropagation, MarkovGP):
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
    def __init__(self, kernel, likelihood, X, Y, R=None, parallel=None):
        super().__init__(kernel, likelihood, X, Y, R=R, parallel=parallel)


class SparseMarkovExpectationPropagationGP(ExpectationPropagation, SparseMarkovGP):
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
    def __init__(self, kernel, likelihood, X, Y, R=None, parallel=None, Z=None):
        super().__init__(kernel, likelihood, X, Y, R=R, parallel=parallel, Z=Z)


# ##### Laplace #####

class LaplaceGP(Laplace, GP):
    """
    [1] Rasmussen, Williams: Gaussian Processes for Machine Learning, 2006
    """
    def __init__(self, kernel, likelihood, X, Y):
        super().__init__(kernel, likelihood, X, Y)


class SparseLaplaceGP(Laplace, SparseGP):
    def __init__(self, kernel, likelihood, X, Y, Z, opt_z=False):
        super().__init__(kernel, likelihood, X, Y, Z, opt_z=opt_z)


class MarkovLaplaceGP(Laplace, MarkovGP):
    def __init__(self, kernel, likelihood, X, Y, R=None, parallel=None):
        super().__init__(kernel, likelihood, X, Y, R=R, parallel=parallel)


class SparseMarkovLaplaceGP(Laplace, SparseMarkovGP):
    def __init__(self, kernel, likelihood, X, Y, R=None, parallel=None, Z=None):
        super().__init__(kernel, likelihood, X, Y, R=R, parallel=parallel, Z=Z)


# ##### Posterior Linearisation #####

class PosteriorLinearisationGP(PosteriorLinearisation, GP):
    """
    [1] Garcia-Fernandez, Tronarp, Sarkka: Gaussian Process Classification
                                           Using Posterior Linearization, IEEE Signal Processing 2019
    """

    def __init__(self, kernel, likelihood, X, Y):
        super().__init__(kernel, likelihood, X, Y)


class SparsePosteriorLinearisationGP(PosteriorLinearisation, SparseGP):
    def __init__(self, kernel, likelihood, X, Y, Z, opt_z=False):
        super().__init__(kernel, likelihood, X, Y, Z, opt_z=opt_z)


class MarkovPosteriorLinearisationGP(PosteriorLinearisation, MarkovGP):
    """
    [1] Garcia-Fernandez, Svensson, Sarkka: Iterated Posterior Linearization Smoother, IEEE Automatic Control 2016
    [2] Wilkinson, Chang, Riis Andersen, Solin: State Space Expectation Propagation, ICML 2020
    """

    def __init__(self, kernel, likelihood, X, Y, R=None, parallel=None):
        super().__init__(kernel, likelihood, X, Y, R=R, parallel=parallel)


class SparseMarkovPosteriorLinearisationGP(PosteriorLinearisation, SparseMarkovGP):
    """
    [1] Wilkinson, Solin, Adam: Sparse Algorithms for Markovian Gaussian Processes, AISTATS 2021
    """

    def __init__(self, kernel, likelihood, X, Y, R=None, parallel=None, Z=None):
        super().__init__(kernel, likelihood, X, Y, R=R, parallel=parallel, Z=Z)


# ##### Taylor #####

class TaylorGP(Taylor, GP):
    """
    [1] Steinberg, Bonilla: Extended and Unscented Gaussian Processes, NeurIPS 2014
    """
    def __init__(self, kernel, likelihood, X, Y):
        super().__init__(kernel, likelihood, X, Y)


class SparseTaylorGP(Taylor, SparseGP):
    def __init__(self, kernel, likelihood, X, Y, Z, opt_z=False):
        super().__init__(kernel, likelihood, X, Y, Z, opt_z=opt_z)


class MarkovTaylorGP(Taylor, MarkovGP):
    """
    [1] Bell: The Iterated Kalman Smoother as a Gauss-Newton method, SIAM Journal on Optimization 1994
    """
    def __init__(self, kernel, likelihood, X, Y, R=None, parallel=None):
        super().__init__(kernel, likelihood, X, Y, R=R, parallel=parallel)


class SparseMarkovTaylorGP(Taylor, SparseMarkovGP):
    def __init__(self, kernel, likelihood, X, Y, R=None, parallel=None, Z=None):
        super().__init__(kernel, likelihood, X, Y, R=R, parallel=parallel, Z=Z)
