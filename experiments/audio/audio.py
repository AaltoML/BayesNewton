import sys
import bayesnewton
import objax
from bayesnewton.cubature import Unscented
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat
import pickle

print('loading data ...')
y = loadmat('speech_female')['y']
fs = 44100  # sampling rate (Hz)
scale = 1000  # convert to milliseconds

normaliser = 0.5 * np.sqrt(np.var(y))
yTrain = y / normaliser  # rescale the data

N = y.shape[0]
x = np.linspace(0., N, num=N) / fs * scale  # arbitrary evenly spaced inputs inputs
# batch_size = 20000
M = 3000
z = np.linspace(x[0], x[-1], num=M)

np.random.seed(123)
# 10-fold cross-validation setup
ind_shuffled = np.random.permutation(N)
ind_split = np.stack(np.split(ind_shuffled, 10))  # 10 random batches of data indices

if len(sys.argv) > 1:
    method = int(sys.argv[1])
    # plot_final = False
    # save_result = True
else:
    method = 2
    # plot_final = True
    # save_result = False

if len(sys.argv) > 2:
    fold = int(sys.argv[2])
else:
    fold = 0

if len(sys.argv) > 3:
    baseline = bool(int(sys.argv[3]))
else:
    baseline = True

if len(sys.argv) > 4:
    parallel = bool(int(sys.argv[4]))
else:
    parallel = None

if len(sys.argv) > 5:
    num_components = int(sys.argv[5])
else:
    num_components = 3

if len(sys.argv) > 6:
    iters = int(sys.argv[6])
else:
    iters = 10

print('method number:', method)
print('batch number:', fold)
print('baseline:', baseline)
print('parallel:', parallel)
print('num components:', num_components)
print('num iterations:', iters)

# Get training and test indices
ind_test = ind_split[fold]  # np.sort(ind_shuffled[:N//10])
ind_train = np.concatenate(ind_split[np.arange(10) != fold])
x_train = x[ind_train]  # 90/10 train/test split
x_test = x[ind_test]
y_train = y[ind_train]
y_test = y[ind_test]

fundamental_freq = 220  # Hz
radial_freq = 2 * np.pi * fundamental_freq / scale  # radial freq = 2pi * f / scale

subband_kernel = bayesnewton.kernels.SubbandMatern12
modulator_kernel = bayesnewton.kernels.Matern52
subband_frequencies = radial_freq * (np.arange(num_components) + 1)
subband_lengthscales = 75. * np.ones(num_components)
modulator_lengthscales = 10. * np.ones(num_components)
modulator_variances = 0.5 * np.ones(num_components)

kern = bayesnewton.kernels.SpectroTemporal(
    subband_lengthscales=subband_lengthscales,
    subband_frequencies=subband_frequencies,
    modulator_lengthscales=modulator_lengthscales,
    modulator_variances=modulator_variances,
    subband_kernel=subband_kernel,
    modulator_kernel=modulator_kernel
)

lik = bayesnewton.likelihoods.AudioAmplitudeDemodulation(num_components=num_components, variance=0.3)


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

unscented_transform = Unscented(dim=None)  # 5th-order unscented transform


lr_adam = 0.05
lr_newton = 0.5
opt_hypers = objax.optimizer.Adam(model.vars())
energy = objax.GradValues(model.energy, model.vars())


@objax.Function.with_vars(model.vars() + opt_hypers.vars())
def train_op():
    model.inference(lr=lr_newton, cubature=unscented_transform)  # perform inference and update variational params
    dE, E = energy(cubature=unscented_transform)  # compute energy and its gradients w.r.t. hypers
    opt_hypers(lr_adam, dE)
    return E


train_op = objax.Jit(train_op)


t0 = time.time()
for i in range(1, iters + 1):
    if i == 2:
        t2 = time.time()
    loss = train_op()
    print('iter %2d, energy: %1.4f' % (i, loss[0]))
t1 = time.time()
t3 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))
print('per-iteration time (excl. compile): %2.2f secs' % ((t3-t2)/(iters-1)))

# calculate posterior predictive distribution via filtering and smoothing at train & test locations:
# print('calculating the posterior predictive distribution ...')
t0 = time.time()
nlpd = model.negative_log_predictive_density(X=x_test, Y=y_test, cubature=unscented_transform)
t1 = time.time()
print('NLPD: %1.2f' % nlpd)
print('prediction time: %2.2f secs' % (t1-t0))

# if save_result:
#     if baseline:
#         with open("output/baseline_" + str(method) + "_" + str(fold) + "_nlpd.txt", "wb") as fp:
#             pickle.dump(nlpd, fp)
#     else:
#         with open("output/" + str(method) + "_" + str(fold) + "_nlpd.txt", "wb") as fp:
#             pickle.dump(nlpd, fp)

# if plot_final:
posterior_mean, posterior_var = model.predict(X=x)
# lb = posterior_mean[:, 0] - np.sqrt(posterior_var[:, 0]) * 1.96
# ub = posterior_mean[:, 0] + np.sqrt(posterior_var[:, 0]) * 1.96

posterior_mean_subbands = posterior_mean[:, :num_components]
posterior_mean_modulators = bayesnewton.utils.softplus(posterior_mean[:, num_components:])
posterior_mean_sig = np.sum(posterior_mean_subbands * posterior_mean_modulators, axis=-1)
posterior_var_subbands = posterior_var[:, :num_components]
posterior_var_modulators = bayesnewton.utils.softplus(posterior_var[:, num_components:])

print('plotting ...')
plt.figure(1, figsize=(12, 5))
plt.clf()
plt.plot(x, y, 'k', label='signal', linewidth=0.6)
plt.plot(x_test, y_test, 'g.', label='test', markersize=4)
plt.plot(x, posterior_mean_sig, 'r', label='posterior mean', linewidth=0.6)
# plt.fill_between(x_pred, lb, ub, color='r', alpha=0.05, label='95% confidence')
plt.xlim(x[0], x[-1])
plt.legend()
plt.title('Audio Signal Processing via Kalman smoothing (human speech signal)')
plt.xlabel('time (milliseconds)')
plt.savefig('fig1.png')

plt.figure(2, figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(x, posterior_mean_subbands, linewidth=0.6)
plt.xlim(x[0], x[-1])
# plt.plot(z, inducing_mean[:, :3, 0], 'r.', label='inducing mean', markersize=4)
plt.title('subbands')
plt.subplot(2, 1, 2)
plt.plot(x, posterior_mean_modulators, linewidth=0.6)
# plt.plot(z, softplus(inducing_mean[:, 3:, 0]), 'r.', label='inducing mean', markersize=4)
plt.xlim(x[0], x[-1])
plt.xlabel('time (milliseconds)')
plt.title('amplitude modulators')
plt.savefig('fig2.png')
plt.show()
