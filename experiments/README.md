# Bayes-Newton Experiments

The code here can be used to reproduce the experiments for the following article:
* W.J. Wilkinson, S. Särkkä, and A. Solin (2021): **Bayes-Newton Methods for Approximate Bayesian Inference with PSD Guarantees**. [*arXiv preprint arXiv:2111.01721*](https://arxiv.org/abs/2111.01721).

The paper experiments can be found in the `motorcycle`, `product` and `gprn` folders respectively. Each folder contains a main Python script, plus bash scripts to produce the results for each inference method class (`bn-newton.sh`, `bn-vi.sh`, `bn-ep.sh`, `bn-pl.sh`). After these have finished running, the `results_bn.py` script can then be run to produce the plots.
