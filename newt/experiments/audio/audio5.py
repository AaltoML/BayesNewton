import sys
import newt
import objax
from newt.cubature import Unscented
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat
import pickle

num_components = 5

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
    fold = int(sys.argv[2])
    plot_final = False
    save_result = True
    iters = 500
else:
    method = 5
    fold = 0
    plot_final = True
    save_result = False
    iters = 150

if len(sys.argv) > 3:
    baseline = int(sys.argv[3])
else:
    baseline = 1

print('method number', method)
print('batch number', fold)

# Get training and test indices
ind_test = ind_split[fold]  # np.sort(ind_shuffled[:N//10])
ind_train = np.concatenate(ind_split[np.arange(10) != fold])
x_train = x[ind_train]  # 90/10 train/test split
x_test = x[ind_test]
y_train = y[ind_train]
y_test = y[ind_test]

fundamental_freq = 220  # Hz
radial_freq = 2 * np.pi * fundamental_freq / scale  # radial freq = 2pi * f / scale
sub1 = newt.kernels.SubbandMatern12(variance=.1, lengthscale=75., radial_frequency=radial_freq, fix_variance=True)
# 1st harmonic
sub2 = newt.kernels.SubbandMatern12(variance=.1, lengthscale=75., radial_frequency=2 * radial_freq, fix_variance=True)
# 2nd harmonic
sub3 = newt.kernels.SubbandMatern12(variance=.1, lengthscale=75., radial_frequency=3 * radial_freq, fix_variance=True)
# 3rd harmonic
sub4 = newt.kernels.SubbandMatern12(variance=.1, lengthscale=75., radial_frequency=4 * radial_freq, fix_variance=True)
# 4th harmonic
sub5 = newt.kernels.SubbandMatern12(variance=.1, lengthscale=75., radial_frequency=5 * radial_freq, fix_variance=True)
# 5th harmonic
sub6 = newt.kernels.SubbandMatern12(variance=.1, lengthscale=75., radial_frequency=6 * radial_freq, fix_variance=True)
# 6th harmonic
sub7 = newt.kernels.SubbandMatern12(variance=.1, lengthscale=75., radial_frequency=7 * radial_freq, fix_variance=True)
mod1 = newt.kernels.Matern52(variance=.5, lengthscale=10., fix_variance=True)
mod2 = newt.kernels.Matern52(variance=.5, lengthscale=10., fix_variance=True)
mod3 = newt.kernels.Matern52(variance=.5, lengthscale=10., fix_variance=True)
mod4 = newt.kernels.Matern52(variance=.5, lengthscale=10., fix_variance=True)
mod5 = newt.kernels.Matern52(variance=.5, lengthscale=10., fix_variance=True)
mod6 = newt.kernels.Matern52(variance=.5, lengthscale=10., fix_variance=True)
mod7 = newt.kernels.Matern52(variance=.5, lengthscale=10., fix_variance=True)

if num_components == 3:
    kern = newt.kernels.Independent([sub1, sub2, sub3,
                                     mod1, mod2, mod3])
elif num_components == 5:
    kern = newt.kernels.Independent([sub1, sub2, sub3, sub4, sub5,
                                     mod1, mod2, mod3, mod4, mod5])
elif num_components == 7:
    kern = newt.kernels.Independent([sub1, sub2, sub3, sub4, sub5, sub6, sub7,
                                     mod1, mod2, mod3, mod4, mod5, mod6, mod7])
else:
    raise NotImplementedError

lik = newt.likelihoods.AudioAmplitudeDemodulation(variance=0.3)

if method == 0:
    inf = newt.inference.Taylor(
        cubature=Unscented(),
        energy_function=newt.inference.ExpectationPropagation(power=0.5, cubature=Unscented()).energy
    )
elif method == 1:
    inf = newt.inference.PosteriorLinearisation(
        cubature=Unscented(),
        energy_function=newt.inference.ExpectationPropagation(power=0.5, cubature=Unscented()).energy
    )
elif method == 2:
    if num_components == 3:
        inf = newt.inference.ExpectationPropagationPSD(power=0.9,
                                                       cubature=Unscented())
        # inf = newt.inference.ExpectationPropagationPSD(power=0.9,
        #                                                cubature=newt.cubature.GaussHermite(num_cub_points=10))
    elif num_components == 5:
        inf = newt.inference.ExpectationPropagationPSD(power=0.9,
                                                       cubature=newt.cubature.GaussHermite(num_cub_points=4))
    elif num_components == 7:
        inf = newt.inference.ExpectationPropagationPSD(power=0.9,
                                                       cubature=newt.cubature.GaussHermite(num_cub_points=3))
    else:
        raise NotImplementedError
elif method == 3:
    if num_components == 3:
        inf = newt.inference.ExpectationPropagationPSD(power=0.5,
                                                       cubature=Unscented())
        # inf = newt.inference.ExpectationPropagationPSD(power=0.9,
        #                                                cubature=newt.cubature.GaussHermite(num_cub_points=10))
    elif num_components == 5:
        inf = newt.inference.ExpectationPropagationPSD(power=0.5,
                                                       cubature=newt.cubature.GaussHermite(num_cub_points=4))
    elif num_components == 7:
        inf = newt.inference.ExpectationPropagationPSD(power=0.5,
                                                       cubature=newt.cubature.GaussHermite(num_cub_points=3))
    else:
        raise NotImplementedError
elif method == 4:
    if num_components == 3:
        inf = newt.inference.ExpectationPropagationPSD(power=0.01,
                                                       cubature=Unscented())
        # inf = newt.inference.ExpectationPropagationPSD(power=0.9,
        #                                                cubature=newt.cubature.GaussHermite(num_cub_points=10))
    elif num_components == 5:
        inf = newt.inference.ExpectationPropagationPSD(power=0.01,
                                                       cubature=newt.cubature.GaussHermite(num_cub_points=4))
    elif num_components == 7:
        inf = newt.inference.ExpectationPropagationPSD(power=0.01, cubature=newt.cubature.GaussHermite(num_cub_points=3))
    else:
        raise NotImplementedError
elif method == 5:
    if num_components == 3:
        inf = newt.inference.VariationalInferencePSD(cubature=Unscented())
    elif num_components == 5:
        inf = newt.inference.VariationalInferencePSD(cubature=newt.cubature.GaussHermite(num_cub_points=4))
    elif num_components == 7:
        inf = newt.inference.VariationalInferencePSD(cubature=newt.cubature.GaussHermite(num_cub_points=3))
    else:
        raise NotImplementedError

if baseline:
    model = newt.models.MarkovGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train)
else:
    model = newt.models.SparseMarkovGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train, Z=z)

trainable_vars = model.vars() + inf.vars()
energy = objax.GradValues(inf.energy, trainable_vars)

lr_adam = 0.05
lr_newton = 0.05
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

# calculate posterior predictive distribution via filtering and smoothing at train & test locations:
# print('calculating the posterior predictive distribution ...')
# t0 = time.time()
# nlpd = model.negative_log_predictive_density(X=x_test, Y=y_test)
# t1 = time.time()
# print('NLPD: %1.2f' % nlpd)
# print('prediction time: %2.2f secs' % (t1-t0))

# if save_result:
#     if baseline:
#         with open("output/baseline_" + str(method) + "_" + str(fold) + "_nlpd.txt", "wb") as fp:
#             pickle.dump(nlpd, fp)
#     else:
#         with open("output/" + str(method) + "_" + str(fold) + "_nlpd.txt", "wb") as fp:
#             pickle.dump(nlpd, fp)

if plot_final:
    posterior_mean, posterior_var = model.predict(X=x)
    # lb = posterior_mean[:, 0] - np.sqrt(posterior_var[:, 0]) * 1.96
    # ub = posterior_mean[:, 0] + np.sqrt(posterior_var[:, 0]) * 1.96

    posterior_mean_subbands = posterior_mean[:, :num_components]
    posterior_mean_modulators = newt.utils.softplus(posterior_mean[:, num_components:])
    posterior_mean_sig = np.sum(posterior_mean_subbands * posterior_mean_modulators, axis=-1)
    posterior_var_subbands = posterior_var[:, :num_components]
    posterior_var_modulators = newt.utils.softplus(posterior_var[:, num_components:])

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
    plt.show()
