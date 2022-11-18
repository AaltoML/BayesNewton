import sys
import bayesnewton
import objax
import numpy as np
import pandas as pd
import time
import pickle

plot_final = False
plot_intermediate = False

print('loading coal data ...')
if plot_final:
    disaster_timings = pd.read_csv('../data/coal.txt', header=None).values[:, 0]
cvind = np.loadtxt('cvind.csv').astype(int)
# 10-fold cross-validation
nt = np.floor(cvind.shape[0]/10).astype(int)
cvind = np.reshape(cvind[:10*nt], (10, nt))

D = np.loadtxt('binned.csv')
x = D[:, 0:1]
y = D[:, 1:]
N = D.shape[0]
N_batch = 300
M = 15
z = np.linspace(np.min(x), np.max(x), M)
num_time_bins = x.shape[0]
binsize = (max(x) - min(x)) / num_time_bins

np.random.seed(123)
# meanval = np.log(len(disaster_timings)/num_time_bins)  # TODO: incorporate mean

if len(sys.argv) > 1:
    method = int(sys.argv[1])
    fold = int(sys.argv[2])
else:
    method = 0
    fold = 0

if len(sys.argv) > 3:
    baseline = bool(int(sys.argv[3]))
else:
    baseline = True

if len(sys.argv) > 4:
    parallel = bool(int(sys.argv[4]))
else:
    parallel = None

print('method number:', method)
print('batch number:', fold)
print('baseline:', baseline)
print('parallel:', parallel)

# Get training and test indices
ind_test = cvind[fold, :]
ind_train = np.setdiff1d(cvind, ind_test)

x_train = x[ind_train, ...]  # 90/10 train/test split
x_test = x[ind_test, ...]
y_train = y[ind_train, ...]
y_test = y[ind_test, ...]

var_f = 1.0  # GP variance
len_f = 4.0  # GP lengthscale

kern = bayesnewton.kernels.Matern52(variance=var_f, lengthscale=len_f)
lik = bayesnewton.likelihoods.Poisson(binsize=binsize)

if baseline:
    if method == 0:
        model = bayesnewton.models.MarkovTaylorGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train, parallel=parallel)
    elif method == 1:
        model = bayesnewton.models.MarkovPosteriorLinearisationGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train,
                                                                  parallel=parallel)
    elif method == 2:
        model = bayesnewton.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train,
                                                       parallel=parallel, power=1.)
    elif method == 3:
        model = bayesnewton.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train,
                                                       parallel=parallel, power=0.5)
    elif method == 4:
        model = bayesnewton.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train,
                                                                  parallel=parallel, power=0.01)
    elif method == 4:
        model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train,
                                                                  parallel=parallel)
else:
    if method == 0:
        model = bayesnewton.models.SparseMarkovTaylorGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train, Z=z, parallel=parallel)
    elif method == 1:
        model = bayesnewton.models.SparseMarkovPosteriorLinearisationGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train, Z=z,
                                                                  parallel=parallel)
    elif method == 2:
        model = bayesnewton.models.SparseMarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train, Z=z,
                                                       parallel=parallel, power=1.)
    elif method == 3:
        model = bayesnewton.models.SparseMarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train, Z=z,
                                                       parallel=parallel, power=0.5)
    elif method == 4:
        model = bayesnewton.models.SparseMarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train, Z=z,
                                                                  parallel=parallel, power=0.01)
    elif method == 4:
        model = bayesnewton.models.SparseMarkovVariationalGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train, Z=z,
                                                                  parallel=parallel)


lr_adam = 0.2
lr_newton = .5
iters = 500
opt_hypers = objax.optimizer.Adam(model.vars())
energy = objax.GradValues(model.energy, model.vars())


@objax.Function.with_vars(model.vars() + opt_hypers.vars())
def train_op():
    model.inference(lr=lr_newton)  # perform inference and update variational params
    dE, E = energy()  # compute energy and its gradients w.r.t. hypers
    opt_hypers(lr_adam, dE)
    return E


train_op = objax.Jit(train_op)

t0 = time.time()
for i in range(1, iters + 1):
    loss = train_op()
    print('iter %2d, energy: %1.4f' % (i, loss[0]))
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

t0 = time.time()
nlpd = model.negative_log_predictive_density(X=x_test, Y=y_test)
t1 = time.time()
print('NLPD: %1.2f' % nlpd)

if baseline:
    with open("output/baseline_" + str(method) + "_" + str(fold) + "_nlpd.txt", "wb") as fp:
        pickle.dump(nlpd, fp)
else:
    with open("output/" + str(method) + "_" + str(fold) + "_nlpd.txt", "wb") as fp:
        pickle.dump(nlpd, fp)
