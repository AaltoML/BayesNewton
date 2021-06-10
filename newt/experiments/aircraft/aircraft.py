import sys
import newt
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

kern_1 = newt.kernels.Matern52(variance=2., lengthscale=scale*5.5e4)
kern_2 = newt.kernels.QuasiPeriodicMatern12(variance=1.,
                                            lengthscale_periodic=scale*2.,
                                            period=scale*365.,
                                            lengthscale_matern=scale*1.5e4)
kern_3 = newt.kernels.QuasiPeriodicMatern12(variance=1.,
                                            lengthscale_periodic=scale*2.,
                                            period=scale*7.,
                                            lengthscale_matern=scale*30*365.)

kern = newt.kernels.Sum([kern_1, kern_2, kern_3])
lik = newt.likelihoods.Poisson(binsize=scale*BIN_WIDTH)

if method == 0:
    inf = newt.inference.Taylor
elif method == 1:
    inf = newt.inference.PosteriorLinearisation
elif method in [2, 3, 4]:
    inf = newt.inference.ExpectationPropagation
elif method == 5:
    inf = newt.inference.VariationalInference

if baseline:
    mod = newt.models.MarkovGP
    Mod = newt.build_model(mod, inf)
    model = Mod(kernel=kern, likelihood=lik, X=x_train, Y=y_train)
else:
    mod = newt.models.SparseMarkovGP
    Mod = newt.build_model(mod, inf)
    model = Mod(kernel=kern, likelihood=lik, X=x_train, Y=y_train, Z=z)

if method == 2:
    inf_args = {"power": 1.}
elif method == 3:
    inf_args = {"power": 0.5}
elif method == 4:
    inf_args = {"power": 0.01}
else:
    inf_args = {}


lr_adam = 0.1
lr_newton = 0.1
iters = 200
opt_hypers = objax.optimizer.Adam(model.vars())
energy = objax.GradValues(model.energy, model.vars())


@objax.Function.with_vars(model.vars() + opt_hypers.vars())
def train_op():
    model.inference(lr=lr_newton, **inf_args)  # perform inference and update variational params
    dE, E = energy(**inf_args)  # compute energy and its gradients w.r.t. hypers
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

# if baseline:
#     with open("output/baseline_" + str(method) + "_" + str(fold) + "_nlpd.txt", "wb") as fp:
#         pickle.dump(nlpd, fp)
# else:
#     with open("output/" + str(method) + "_" + str(fold) + "_nlpd.txt", "wb") as fp:
#         pickle.dump(nlpd, fp)

# with open("output/" + str(method) + "_" + str(fold) + "_nlpd.txt", "rb") as fp:
#     nlpd_show = pickle.load(fp)
# print(nlpd_show)

# plt.figure(1)
# plt.plot(t_test, mu, 'b-')
# plt.plot(z, inducing_mean[..., 0], 'b.', label='inducing mean', markersize=8)
# plt.show()
