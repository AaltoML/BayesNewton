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

Newt is a Gaussian process library built in [JAX](https://github.com/google/jax) (with [objax](https://github.com/google/objax)).

Newt currently provides the following models:
 - GPs
 - Sparse GPs
 - Markov GPs
 - Sparse Markov GPs

with the following inference methods:
 - Variational inference (with natural gradients)
 - Power expectation propagation
 - Laplace
 - Posterior linearisation (i.e. classical nonlinear Kalman smoothers)
 - Taylor (i.e. analytical linearisation / extended Kalman smoother)

## Installation
In the top directory (Newt), run
```bash
pip install -e .
```

## License

This software is provided under the Apache License 2.0. See the accompanying LICENSE file for details.
