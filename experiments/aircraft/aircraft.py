import sys
import bayesnewton
import objax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import date
import pickle

plot_final = False
plot_intermediate = False

print('loading data ...')
aircraft_accidents = pd.read_csv('aircraft_accidents.txt', sep='-', header=None).values

num_data = aircraft_accidents.shape[0]
xx = np.zeros([num_data, 1])
for j in range(num_data):
    xx[j] = date.toordinal(date(aircraft_accidents[j, 0], aircraft_accidents[j, 1], aircraft_accidents[j, 2])) + 366

BIN_WIDTH = 1
# Discretize the data
x_min = np.floor(np.min(xx))
x_max = np.ceil(np.max(xx))
x_max_int = x_max-np.mod(x_max-x_min, BIN_WIDTH)
x = np.linspace(x_min, x_max_int, num=int((x_max_int-x_min)/BIN_WIDTH+1))
x = np.concatenate([np.min(x)-np.linspace(61, 1, num=61), x])  # pad with zeros to reduce strange edge effects
y, _ = np.histogram(xx, np.concatenate([[-1e10], x[1:]-np.diff(x)/2, [1e10]]))
N = y.shape[0]
print('N =', N)

scale = 1  # scale inputs for stability
x = scale * x

np.random.seed(123)
ind_shuffled = np.random.permutation(N)
ind_split = np.stack(np.split(ind_shuffled, 10))  # 10 random batches of data indices

np.random.seed(123)
# meanval = np.log(len(disaster_timings)/num_time_bins)  # TODO: incorporate mean

if len(sys.argv) > 1:
    method = int(sys.argv[1])
    fold = int(sys.argv[2])
else:
    method = 2
    fold = 0

print('method number', method)
print('batch number', fold)

# Get training and test indices
ind_test = ind_split[fold]  # np.sort(ind_shuffled[:N//10])
ind_train = np.concatenate(ind_split[np.arange(10) != fold])
x_train = x[ind_train]  # 90/10 train/test split
x_test = x[ind_test]
y_train = y[ind_train]
y_test = y[ind_test]
# N_batch = 2000
M = 4000
# z = np.linspace(701050, 737050, M)
z = np.linspace(x[0], x[-1], M)

if len(sys.argv) > 3:
    baseline = int(sys.argv[3])
else:
    baseline = 0

kern_1 = bayesnewton.kernels.Matern52(variance=2., lengthscale=scale*5.5e4)
kern_2 = bayesnewton.kernels.QuasiPeriodicMatern12(variance=1.,
                                                   lengthscale_periodic=scale*2.,
                                                   period=scale*365.,
                                                   lengthscale_matern=scale*1.5e4)
kern_3 = bayesnewton.kernels.QuasiPeriodicMatern12(variance=1.,
                                                   lengthscale_periodic=scale*2.,
                                                   period=scale*7.,
                                                   lengthscale_matern=scale*30*365.)

kern = bayesnewton.kernels.Sum([kern_1, kern_2, kern_3])
lik = bayesnewton.likelihoods.Poisson(binsize=scale*BIN_WIDTH)

if baseline:
    if method == 0:
        model = bayesnewton.models.MarkovTaylorGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train)
    elif method == 1:
        model = bayesnewton.models.MarkovPosteriorLinearisationGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train)
    elif method == 2:
        model = bayesnewton.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train, power=1.)
    elif method == 3:
        model = bayesnewton.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train, power=0.5)
    elif method == 4:
        model = bayesnewton.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train, power=0.01)
    elif method == 4:
        model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train)
else:
    if method == 0:
        model = bayesnewton.models.SparseMarkovTaylorGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train, Z=z)
    elif method == 1:
        model = bayesnewton.models.SparseMarkovPosteriorLinearisationGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train, Z=z)
    elif method == 2:
        model = bayesnewton.models.SparseMarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train, Z=z, power=1.)
    elif method == 3:
        model = bayesnewton.models.SparseMarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train, Z=z, power=0.5)
    elif method == 4:
        model = bayesnewton.models.SparseMarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train, Z=z, power=0.01)
    elif method == 4:
        model = bayesnewton.models.SparseMarkovVariationalGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train, Z=z)


lr_adam = 0.1
lr_newton = 0.1
iters = 200
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

# calculate posterior predictive distribution via filtering and smoothing at train & test locations:
print('calculating the posterior predictive distribution ...')
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

# with open("output/" + str(method) + "_" + str(fold) + "_nlpd.txt", "rb") as fp:
#     nlpd_show = pickle.load(fp)
# print(nlpd_show)

# plt.figure(1)
# plt.plot(t_test, mu, 'b-')
# plt.plot(z, inducing_mean[..., 0], 'b.', label='inducing mean', markersize=8)
# plt.show()
