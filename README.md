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

Newt is a Gaussian process (GP) library built in [JAX](https://github.com/google/jax) (with [objax](https://github.com/google/objax)). 

Newt provides a unifying view of approximate Bayesian inference for GPs, and allows for the combination of many models (e.g. GPs, sparse GPs, Markov GPs, sparse Markov GPs) with the inference method of your choice (VI, EP, Laplace, Linearisation). For a full list of the methods implemented see the [models](https://github.com/AaltoML/Newt/blob/main/newt/models.py) file.

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
