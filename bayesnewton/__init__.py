from . import (
    kernels,
    utils,
    ops,
    likelihoods,
    models,
    basemodels,
    inference,
    cubature
)


def build_model(model, inf, name='GPModel'):
    return type(name, (inf, model), {})
