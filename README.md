# Bayes-Newton

Bayes-Newton is a library for approximate inference in Gaussian processes (GPs) in [JAX](https://github.com/google/jax) (with [objax](https://github.com/google/objax)), built and actively maintained by [Will Wilkinson](https://wil-j-wil.github.io/).

Bayes-Newton provides a unifying view of approximate Bayesian inference, and allows for the combination of many models (e.g. GPs, sparse GPs, Markov GPs, sparse Markov GPs) with the inference method of your choice (VI, EP, Laplace, Linearisation). For a full list of the methods implemented scroll down to the bottom of this page.

The methodology is outlined in the following article:
* W.J. Wilkinson, S. Särkkä, and A. Solin (2021): **Bayes-Newton Methods for Approximate Bayesian Inference with PSD Guarantees**. [*arXiv preprint arXiv:2111.01721*](https://arxiv.org/abs/2111.01721).

## Installation
```bash
pip install bayesnewton
```

## Example
Given some inputs `x` and some data `y`, you can construct a Bayes-Newton model as follows,
```python
kern = bayesnewton.kernels.Matern52()
lik = bayesnewton.likelihoods.Gaussian()
model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=x, Y=y)
```
The training loop (inference and hyperparameter learning) is then set up using objax's built in functionality:
```python
lr_adam = 0.1
lr_newton = 1
opt_hypers = objax.optimizer.Adam(model.vars())
energy = objax.GradValues(model.energy, model.vars())

@objax.Function.with_vars(model.vars() + opt_hypers.vars())
def train_op():
    model.inference(lr=lr_newton, **inf_args)  # perform inference and update variational params
    dE, E = energy(**inf_args)  # compute energy and its gradients w.r.t. hypers
    opt_hypers(lr_adam, dE)  # update the hyperparameters
    return E
```
As we are using JAX, we can JIT compile the training loop:
```python
train_op = objax.Jit(train_op)
```
and then run the training loop,
```python
iters = 20
for i in range(1, iters + 1):
    loss = train_op()
```
Full demos are available [here](https://github.com/AaltoML/BayesNewton/tree/main/demos).

## Citing Bayes-Newton

```
@article{wilkinson2021bayesnewton,
  title = {{B}ayes-{N}ewton Methods for Approximate {B}ayesian Inference with {PSD} Guarantees},
  author = {Wilkinson, William J. and S\"arkk\"a, Simo and Solin, Arno},
  journal={arXiv preprint arXiv:2111.01721},
  year={2021}
}
```

## Implemented Models
For a full list of the all the models available see the [model class list](https://github.com/AaltoML/BayesNewton/blob/main/bayesnewton/models.py).

### Variational GPs
 - **Variationl GP** *(Opper, Archambeau: The Variational Gaussian Approximation Revisited, Neural Computation 2009; Khan, Lin: Conugate-Computation Variational Inference - Converting Inference in Non-Conjugate Models in to Inference in Conjugate Models, AISTATS 2017)*
 - **Sparse Variational GP** *(Hensman, Matthews, Ghahramani: Scalable Variational Gaussian Process Classification, AISTATS 2015; Adam, Chang, Khan, Solin: Dual Parameterization of Sparse Variational Gaussian Processes, NeurIPS 2021)*
 - **Markov Variational GP** *(Chang, Wilkinson, Khan, Solin: Fast Variational Learning in State Space Gaussian Process Models, MLSP 2020)*
 - **Sparse Markov Variational GP** *(Adam, Eleftheriadis, Durrande, Artemev, Hensman: Doubly Sparse Variational Gaussian Processes, AISTATS 2020; Wilkinson, Solin, Adam: Sparse Algorithms for Markovian Gaussian Processes, AISTATS 2021)*
 - **Spatio-Temporal Variational GP** *(Hamelijnck, Wilkinson, Loppi, Solin, Damoulas: Spatio-Temporal Variational Gaussian Processes, NeurIPS 2021)*
### Expectation Propagation GPs
 - **Expectation Propagation GP** *(Minka: A Family of Algorithms for Approximate Bayesian Inference, Ph. D thesis 2000)*
 - **Sparse Expectation Propagation GP (energy not working)** *(Csato, Opper: Sparse on-line Gaussian processes, Neural Computation 2002; Bui, Yan, Turner: A Unifying Framework for Gaussian Process Pseudo Point Approximations Using Power Expectation Propagation, JMLR 2017)*
 - **Markov Expectation Propagation GP** *(Wilkinson, Chang, Riis Andersen, Solin: State Space Expectation Propagation, ICML 2020)*
 - **Sparse Markov Expectation Propagation GP** *(Wilkinson, Solin, Adam: Sparse Algorithms for Markovian Gaussian Processes, AISTATS 2021)*
### Laplace/Newton GPs
 - **Laplace GP** *(Rasmussen, Williams: Gaussian Processes for Machine Learning, 2006)*
 - **Sparse Laplace GP** *(Wilkinson, Särkkä, Solin: Bayes-Newton Methods for Approximate Bayesian Inference with PSD Guarantees)*
 - **Markov Laplace GP** *(Wilkinson, Särkkä, Solin: Bayes-Newton Methods for Approximate Bayesian Inference with PSD Guarantees)*
 - **Sparse Markov Laplace GP**
### Linearisation GPs
 - **Posterior Linearisation GP** *(García-Fernández, Tronarp, Sarkka: Gaussian Process Classification Using Posterior Linearization, IEEE Signal Processing 2019; Steinberg, Bonilla: Extended and Unscented Gaussian Processes, NeurIPS 2014)*
 - **Sparse Posterior Linearisation GP**
 - **Markov Posterior Linearisation GP** *(García-Fernández, Svensson, Sarkka: Iterated Posterior Linearization Smoother, IEEE Automatic Control 2016; Wilkinson, Chang, Riis Andersen, Solin: State Space Expectation Propagation, ICML 2020)*
 - **Sparse Markov Posterior Linearisation GP** *(Wilkinson, Solin, Adam: Sparse Algorithms for Markovian Gaussian Processes, AISTATS 2021)*
 - **Taylor Expansion / Analytical Linearisaiton GP** *(Steinberg, Bonilla: Extended and Unscented Gaussian Processes, NeurIPS 2014)*
 - **Markov Taylor GP / Extended Kalman Smoother** *(Bell: The Iterated Kalman Smoother as a Gauss-Newton method, SIAM Journal on Optimization 1994)*
 - **Sparse Taylor GP**
 - **Sparse Markov Taylor GP / Sparse Extended Kalman Smoother** *(Wilkinson, Solin, Adam: Sparse Algorithms for Markovian Gaussian Processes, AISTATS 2021)*

## Gauss-Newton GPs
*(Wilkinson, Särkkä, Solin: Bayes-Newton Methods for Approximate Bayesian Inference with PSD Guarantees)*
 - **Gauss-Newton** 
 - **Variational Gauss-Newton**
 - **PEP Gauss-Newton**
 - **2nd-order PL Gauss-Newton**

## Quasi-Newton GPs
*(Wilkinson, Särkkä, Solin: Bayes-Newton Methods for Approximate Bayesian Inference with PSD Guarantees)*
 - **Quasi-Newton** 
 - **Variational Quasi-Newton**
 - **PEP Quasi-Newton**
 - **PL Quasi-Newton**

## GPs with PSD Constraints via Riemannian Gradients
 - **VI Riemann Grad** *(Lin, Schmidt, Khan: Handling the Positive-Definite Constraint in the Bayesian Learning Rule, ICML 2020)*
 - **Newton/Laplace Riemann Grad** *(Lin, Schmidt, Khan: Handling the Positive-Definite Constraint in the Bayesian Learning Rule, ICML 2020)*
 - **PEP Riemann Grad** *(Wilkinson, Särkkä, Solin: Bayes-Newton Methods for Approximate Bayesian Inference with PSD Guarantees)*

## Others

 - **Infinite Horizon GP** *(Solin, Hensman, Turner: Infinite-Horizon Gaussian Processes, NeurIPS 2018)*
 - **Parallel Markov GP (with VI, EP, PL, ...)** *(Särkkä, García-Fernández: Temporal parallelization of Bayesian smoothers; Corenflos, Zhao, Särkkä: Gaussian Process Regression in Logarithmic Time; Hamelijnck, Wilkinson, Loppi, Solin, Damoulas: Spatio-Temporal Variational Gaussian Processes, NeurIPS 2021)*
 - **2nd-order Posterior Linearisation GP (sparse, Markov, ...)** *(Wilkinson, Särkkä, Solin: Bayes-Newton Methods for Approximate Bayesian Inference with PSD Guarantees)*

## License

This software is provided under the Apache License 2.0. See the accompanying LICENSE file for details.
