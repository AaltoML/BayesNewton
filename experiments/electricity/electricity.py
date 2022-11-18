import sys
import bayesnewton
import objax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle

plot_intermediate = False

print('loading data ...')
np.random.seed(99)
# N = 52 * 10080  # 10080 = one week, 2049280 total points
N = 26 * 10080  # 6 months
electricity_data = pd.read_csv('./electricity.csv', sep='  ', header=None, engine='python').values[:N, :]
x = electricity_data[:, 0][:, None]
y = electricity_data[:, 1][:, None]
print('N =', N)

ind_shuffled = np.random.permutation(N)
ind_split = np.stack(np.split(ind_shuffled, 10))  # 10 random batches of data indices

if len(sys.argv) > 1:
    plot_final = False
    method = int(sys.argv[1])
    fold = int(sys.argv[2])
else:
    plot_final = True
    method = 4
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
ind_test = ind_split[fold]  # np.sort(ind_shuffled[:N//10])
ind_train = np.concatenate(ind_split[np.arange(10) != fold])

x_train = x[ind_train]  # 90/10 train/test split
x_test = x[ind_test]
y_train = y[ind_train]
y_test = y[ind_test]
# batch_size = 100000  # x_train.shape[0]
batch_size = 50000  # x_train.shape[0]
# M = 100000
M = 50000
z = np.linspace(x[0], x[-1], M)

var_y = .1
var_f = 1.  # GP variance
len_f = 1.  # GP lengthscale
period = 1.  # period of quasi-periodic component
len_p = 5.  # lengthscale of quasi-periodic component
var_f_mat = 1.
len_f_mat = 1.

kern1 = bayesnewton.kernels.Matern32(variance=var_f_mat, lengthscale=len_f_mat)
kern2 = bayesnewton.kernels.QuasiPeriodicMatern12(variance=var_f,
                                           lengthscale_periodic=len_p,
                                           period=period,
                                           lengthscale_matern=len_f)
kern = bayesnewton.kernels.Sum([kern1, kern2])

lik = bayesnewton.likelihoods.Gaussian(variance=var_y)

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


# if plot_final:
#     lb = posterior_mean[:, 0, 0] - 1.96 * posterior_cov[:, 0, 0]**0.5
#     ub = posterior_mean[:, 0, 0] + 1.96 * posterior_cov[:, 0, 0]**0.5
#     t_test = model.t_all[model.test_id, 0]
#
#     print('plotting ...')
#     plt.figure(1, figsize=(12, 5))
#     plt.clf()
#     plt.plot(x, y, 'b.', label='training observations', markersize=4)
#     plt.plot(x_test, y_test, 'r.', alpha=0.5, label='test observations', markersize=4)
#     plt.plot(t_test, posterior_mean[:, 0], 'g', label='posterior mean')
#     plt.plot(z, inducing_mean[:, 0], 'g.', label='inducing mean')
#     plt.fill_between(t_test, lb, ub, color='g', alpha=0.05, label='95% confidence')
#     plt.xlim(t_test[0], t_test[-1])
#     plt.legend()
#     plt.title('GP regression via Kalman smoothing. Test NLPD: %1.2f' % nlpd)
#     plt.xlabel('time, $t$')
#     plt.show()
