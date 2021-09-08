# Newt

                          __ \/_
                         (' \`\
                      _\, \ \\/ 
                       /`\/\ \\
                            \ \\    
                             \ \\/\/_
                             /\ \\'\
                           __\ `\\\
                            /|`  `\\
                                   \\
                                    \\
                                     \\    ,
                                      `---'  

Newt is a Gaussian process (GP) library built in [JAX](https://github.com/google/jax) (with [objax](https://github.com/google/objax)), built and actively maintained by [Will Wilkinson](https://wil-j-wil.github.io/).

Newt provides a unifying view of approximate Bayesian inference for GPs, and allows for the combination of many models (e.g. GPs, sparse GPs, Markov GPs, sparse Markov GPs) with the inference method of your choice (VI, EP, Laplace, Linearisation). For a full list of the methods implemented scroll down to the bottom of this page.

## Installation
In the top directory (Newt), run
```bash
pip install -e .
```

## Example
Given some inputs `x` and some data `y`, you can construct a Newt model as follows,
```python
kern = newt.kernels.Matern52()
lik = newt.likelihoods.Gaussian()
model = newt.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=x, Y=y)
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
Full demos are available [here](https://github.com/AaltoML/Newt/tree/main/newt/demos).

## License

This software is provided under the Apache License 2.0. See the accompanying LICENSE file for details.

## Citing Newt

```
@software{newt2021github,
  author = {William J. Wilkinson},
  title = {{Newt}},
  url = {https://github.com/AaltoML/Newt},
  version = {0.0},
  year = {2021},
}
```

## Implemented Models

### Variational GPs
 - **Variationl GP** *(Opper, Archambeau: The Variational Gaussian Approximation Revisited, Neural Computation 2009; Khan, Lin: Conugate-Computation Variational Inference - Converting Inference in Non-Conjugate Models in to Inference in Conjugate Models, AISTATS 2017)*
 - **Sparse Variational GP** *(Hensman, Matthews, Ghahramani: Scalable Variational Gaussian Process Classification, AISTATS 2015)*
 - **Markov Variational GP** *(Chang, Wilkinson, Khan, Solin: Fast Variational Learning in State Space Gaussian Process Models, MLSP 2020)*
 - **Sparse Markov Variational GP** *(Adam, Eleftheriadis, Durrande, Artemev, Hensman: Doubly Sparse Variational Gaussian Processes, AISTATS 2020; Wilkinson, Solin, Adam: Sparse Algorithms for Markovian Gaussian Processes, AISTATS 2021)*
### Expectation Propagation GPs
 - **Expectation Propagation GP** *(Minka: A Family of Algorithms for Approximate Bayesian Inference, Ph. D thesis 2000)*
 - **Sparse Expectation Propagation GP (energy not working)** *(Csato, Opper: Sparse on-line Gaussian processes, Neural Computation 2002; Bui, Yan, Turner: A Unifying Framework for Gaussian Process Pseudo Point Approximations Using Power Expectation Propagation, JMLR 2017)*
 - **Markov Expectation Propagation GP** *(Wilkinson, Chang, Riis Andersen, Solin: State Space Expectation Propagation, ICML 2020)*
 - **Sparse Markov Expectation Propagation GP** *(Wilkinson, Solin, Adam: Sparse Algorithms for Markovian Gaussian Processes, AISTATS 2021)*
### Laplace GPs
 - **Laplace GP** *(Rasmussen, Williams: Gaussian Processes for Machine Learning, 2006)*
 - **Sparse Laplace GP**
 - **Markov Laplace GP**
 - **Sparse Markov Laplace GP**
### Linearisation GPs
 - **Posterior Linearisation GP** *(Garcia-Fernandez, Tronarp, Sarkka: Gaussian Process Classification Using Posterior Linearization, IEEE Signal Processing 2019; Steinberg, Bonilla: Extended and Unscented Gaussian Processes, NeurIPS 2014)*
 - **Sparse Posterior Linearisation GP**
 - **Markov Posterior Linearisation GP** *(Garcia-Fernandez, Svensson, Sarkka: Iterated Posterior Linearization Smoother, IEEE Automatic Control 2016; Wilkinson, Chang, Riis Andersen, Solin: State Space Expectation Propagation, ICML 2020)*
 - **Sparse Markov Posterior Linearisation GP** *(Wilkinson, Solin, Adam: Sparse Algorithms for Markovian Gaussian Processes, AISTATS 2021)*
 - **Taylor Expansion / Analytical Linearisaiton GP** *(Steinberg, Bonilla: Extended and Unscented Gaussian Processes, NeurIPS 2014)*
 - **Markov Taylor GP / Extended Kalman Smoother** *(Bell: The Iterated Kalman Smoother as a Gauss-Newton method, SIAM Journal on Optimization 1994)*
 - **Sparse Taylor GP**
 - **Sparse Markov Taylor GP / Sparse Extended Kalman Smoother** *(Wilkinson, Solin, Adam: Sparse Algorithms for Markovian Gaussian Processes, AISTATS 2021)*
