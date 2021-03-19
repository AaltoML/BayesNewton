import sys
import newt
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
    baseline = int(sys.argv[3])
else:
    baseline = 0

print('method number', method)
print('batch number', fold)

# Get training and test indices
ind_test = cvind[fold, :]
ind_train = np.setdiff1d(cvind, ind_test)

x_train = x[ind_train, ...]  # 90/10 train/test split
x_test = x[ind_test, ...]
y_train = y[ind_train, ...]
y_test = y[ind_test, ...]

var_f = 1.0  # GP variance
len_f = 4.0  # GP lengthscale

kern = newt.kernels.Matern52(variance=var_f, lengthscale=len_f)
lik = newt.likelihoods.Poisson(binsize=binsize)

if method == 0:
    inf = newt.inference.Taylor()
elif method == 1:
    inf = newt.inference.PosteriorLinearisation()
elif method == 2:
    inf = newt.inference.ExpectationPropagation(power=1)
elif method == 3:
    inf = newt.inference.ExpectationPropagation(power=0.5)
elif method == 4:
    inf = newt.inference.ExpectationPropagation(power=0.01)
elif method == 5:
    inf = newt.inference.VariationalInference()

if baseline:
    model = newt.models.MarkovGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train)
else:
    model = newt.models.SparseMarkovGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train, Z=z)

trainable_vars = model.vars() + inf.vars()
energy = objax.GradValues(inf.energy, trainable_vars)

lr_adam = 0.2
lr_newton = .5
iters = 500
opt = objax.optimizer.Adam(trainable_vars)


def train_op():
    inf(model, lr=lr_newton)  # perform inference and update variational params
    dE, E = energy(model)  # compute energy and its gradients w.r.t. hypers
    return dE, E


train_op = objax.Jit(train_op, trainable_vars)

t0 = time.time()
for i in range(1, iters + 1):
    grad, loss = train_op()
    opt(lr_adam, grad)
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
