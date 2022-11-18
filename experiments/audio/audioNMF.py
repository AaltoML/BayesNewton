import sys
import bayesnewton
import objax
from bayesnewton.cubature import Unscented
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat

print('loading data ...')
y_raw = loadmat('speech_female')['y']
fs = 44100  # sampling rate (Hz)
scale_x = 1000  # convert to milliseconds
scale_y = 1.  # scale signal up to deal with Gauss-Newton instability at low obs noise

# normaliser = 0.5 * np.sqrt(np.var(y_raw))
# y = y_raw / normaliser * scale_y  # rescale the data
y = y_raw * scale_y  # rescale the data

N = y.shape[0]
x = np.linspace(0., N, num=N) / fs * scale_x  # arbitrary evenly spaced inputs
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
    method = 13
    # plot_final = True
    # save_result = False

if len(sys.argv) > 2:
    fold = int(sys.argv[2])
else:
    fold = 0

if len(sys.argv) > 3:
    parallel = bool(int(sys.argv[3]))
else:
    parallel = None

if len(sys.argv) > 4:
    num_subbands = int(sys.argv[4])
else:
    num_subbands = 6

if len(sys.argv) > 5:
    num_modulators = int(sys.argv[5])
else:
    num_modulators = 2

if len(sys.argv) > 6:
    iters = int(sys.argv[6])
else:
    iters = 200

print('method number:', method)
print('batch number:', fold)
print('parallel:', parallel)
print('num subbands:', num_subbands)
print('num modulators:', num_modulators)
print('num iterations:', iters)

# Get training and test indices
ind_test = ind_split[fold]  # np.sort(ind_shuffled[:N//10])
ind_train = np.concatenate(ind_split[np.arange(10) != fold])
x_train = x[ind_train]  # 90/10 train/test split
x_test = x[ind_test]
y_train = y[ind_train]
y_test = y[ind_test]

fundamental_freq = 220  # Hz
# radial_freq = 2 * np.pi * fundamental_freq / scale  # radial freq = 2pi * f / scale

subband_kernel = bayesnewton.kernels.SubbandMatern32
modulator_kernel = bayesnewton.kernels.Matern52
subband_frequencies = fundamental_freq / scale_x * (np.arange(num_subbands) + 1)
subband_lengthscales = 75. * np.ones(num_subbands)
modulator_lengthscales = 10. * np.ones(num_modulators)
modulator_variances = 0.5 * np.ones(num_modulators) * scale_y

kern = bayesnewton.kernels.SpectroTemporal(
    subband_lengthscales=subband_lengthscales,
    subband_frequencies=subband_frequencies,
    modulator_lengthscales=modulator_lengthscales,
    modulator_variances=modulator_variances,
    subband_kernel=subband_kernel,
    modulator_kernel=modulator_kernel
)

lik = bayesnewton.likelihoods.NonnegativeMatrixFactorisation(
    num_subbands=num_subbands,
    num_modulators=num_modulators,
    variance=0.17 * scale_y,
    # fix_variance=True
)

if method == 0:
    model = bayesnewton.models.MarkovTaylorGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train, parallel=parallel)
elif method == 1:
    model = bayesnewton.models.MarkovPosteriorLinearisationGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train,
                                                       parallel=parallel)
elif method == 2:
    # model = bayesnewton.models.MarkovExpectationPropagationGaussNewtonGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train,
    #                                                               parallel=parallel, power=1.)
    model = bayesnewton.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train,
                                                       parallel=parallel, power=1.)
elif method == 3:
    # model = bayesnewton.models.MarkovExpectationPropagationGaussNewtonGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train,
    #                                                               parallel=parallel, power=0.5)
    model = bayesnewton.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train,
                                                       parallel=parallel, power=0.5)
elif method == 4:
    # model = bayesnewton.models.MarkovExpectationPropagationGaussNewtonGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train,
    #                                                               parallel=parallel, power=0.01)
    model = bayesnewton.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train,
                                                       parallel=parallel, power=0.01)
elif method == 5:
    model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train,
                                            parallel=parallel)
elif method == 6:
    model = bayesnewton.models.MarkovLaplaceGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train, parallel=parallel)
elif method == 7:
    model = bayesnewton.models.MarkovPosteriorLinearisation2ndOrderGaussNewtonGP(kernel=kern, likelihood=lik, X=x_train,
                                                                                 Y=y_train, parallel=parallel)
elif method == 8:
    model = bayesnewton.models.MarkovVariationalGaussNewtonGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train,
                                                       parallel=parallel)
# elif method == 9:
#     model = bayesnewton.models.MarkovExpectationPropagationGaussNewtonGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train,
#                                                                   parallel=parallel, power=1.)
elif method == 10:
    model = bayesnewton.models.MarkovVGNEPGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train, parallel=parallel)
elif method == 11:
    model = bayesnewton.models.MarkovGaussNewtonGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train, parallel=parallel)
elif method == 12:
    model = bayesnewton.models.MarkovQuasiNewtonGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train,
                                            parallel=parallel)
elif method == 13:
    model = bayesnewton.models.MarkovVariationalQuasiNewtonGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train,
                                                       parallel=parallel)
elif method == 14:
    model = bayesnewton.models.MarkovExpectationPropagationQuasiNewtonGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train,
                                                                  parallel=parallel, power=0.5)
elif method == 15:
    model = bayesnewton.models.MarkovPosteriorLinearisation2ndOrderQuasiNewtonGP(kernel=kern, likelihood=lik, X=x_train,
                                                                                 Y=y_train, parallel=parallel)
# elif method == 16:
#     model = bayesnewton.models.MarkovPosteriorLinearisationQuasiNewtonGP(kernel=kern, likelihood=lik, X=x_train,
#                                                                   Y=y_train, parallel=parallel)
elif method == 17:
    model = bayesnewton.models.MarkovVariationalRiemannGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train, parallel=parallel)
elif method == 18:
    model = bayesnewton.models.MarkovExpectationPropagationRiemannGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train,
                                                                     parallel=parallel)
elif method == 19:
    model = bayesnewton.models.MarkovExpectationPropagationRiemannGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train,
                                                                     parallel=parallel, power=0.5)
print('model:', model)

# unscented_transform = Unscented(dim=num_modulators)  # 5th-order unscented transform
# unscented_transform = Unscented(dim=num_modulators+num_subbands)  # 5th-order unscented transform
unscented_transform = Unscented(dim=None)  # 5th-order unscented transform

lr_adam = 0.05
lr_newton = 0.3
opt_hypers = objax.optimizer.Adam(model.vars())
energy = objax.GradValues(model.energy, model.vars())

damping = np.logspace(np.log10(1.), np.log10(1e-2), num=iters)


@objax.Function.with_vars(model.vars() + opt_hypers.vars())
def train_op(damp):
    model.inference(lr=lr_newton, cubature=unscented_transform, damping=damp)  # perform inference and update variational params
    dE, E = energy(cubature=unscented_transform)  # compute energy and its gradients w.r.t. hypers
    # opt_hypers(lr_adam, dE)
    test_nlpd_ = model.negative_log_predictive_density(X=x_test, Y=y_test, cubature=unscented_transform)
    return E, test_nlpd_


train_op = objax.Jit(train_op)


t0 = time.time()
for i in range(1, iters + 1):
    if i == 2:
        t2 = time.time()
    loss, test_nlpd = train_op(damping[i-1])
    print('iter %2d, energy: %1.4f, nlpd: %1.4f' % (i, loss[0], test_nlpd))
    print(model.likelihood.variance)
    # print(
    #     'lengthscales: ',
    #     model.kernel.kernel0.lengthscale,
    #     model.kernel.kernel1.lengthscale,
    #     model.kernel.kernel2.lengthscale,
    #     model.kernel.kernel3.lengthscale,
    #     model.kernel.kernel4.lengthscale,
    #     model.kernel.kernel5.lengthscale,
    #     model.kernel.kernel6.lengthscale,
    #     model.kernel.kernel7.lengthscale,
    #     model.kernel.kernel8.lengthscale,
    #     model.kernel.kernel9.lengthscale,
    #     model.kernel.kernel10.lengthscale,
    #     model.kernel.kernel11.lengthscale,
    #     model.kernel.kernel12.lengthscale,
    #     model.kernel.kernel13.lengthscale,
    #     model.kernel.kernel14.lengthscale,
    #     model.kernel.kernel15.lengthscale,
    # )
    # print(
    #     'variances: ',
    #     model.kernel.kernel0.variance,
    #     model.kernel.kernel1.variance,
    #     model.kernel.kernel2.variance,
    #     model.kernel.kernel3.variance,
    #     model.kernel.kernel4.variance,
    #     model.kernel.kernel5.variance,
    #     model.kernel.kernel6.variance,
    #     model.kernel.kernel7.variance,
    #     model.kernel.kernel8.variance,
    #     model.kernel.kernel9.variance,
    #     model.kernel.kernel10.variance,
    #     model.kernel.kernel11.variance,
    #     model.kernel.kernel12.variance,
    #     model.kernel.kernel13.variance,
    #     model.kernel.kernel14.variance,
    #     model.kernel.kernel15.variance,
    # )
    # print(
    #     'radial freqs.: ',
    #     model.kernel.kernel0.radial_frequency,
    #     model.kernel.kernel1.radial_frequency,
    #     model.kernel.kernel2.radial_frequency,
    #     model.kernel.kernel3.radial_frequency,
    #     model.kernel.kernel4.radial_frequency,
    #     model.kernel.kernel5.radial_frequency,
    #     model.kernel.kernel6.radial_frequency,
    #     model.kernel.kernel7.radial_frequency,
    #     model.kernel.kernel8.radial_frequency,
    #     model.kernel.kernel9.radial_frequency,
    #     model.kernel.kernel10.radial_frequency,
    #     model.kernel.kernel11.radial_frequency,
    # )
    # print('weights: ', model.likelihood.weights)
    # print('lik. variance: ', model.likelihood.variance)
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

# if plot_final:
posterior_mean, posterior_var = model.predict(X=x)
# lb = posterior_mean[:, 0] - np.sqrt(posterior_var[:, 0]) * 1.96
# ub = posterior_mean[:, 0] + np.sqrt(posterior_var[:, 0]) * 1.96

posterior_mean_subbands = posterior_mean[:, :num_subbands]
posterior_mean_modulators = bayesnewton.utils.softplus(posterior_mean[:, num_subbands:])
posterior_mean_sig = np.sum(
    posterior_mean_subbands * (model.likelihood.weights[None] @ posterior_mean_modulators[..., None])[..., 0],
    axis=-1
)
posterior_var_subbands = posterior_var[:, :num_subbands]
posterior_var_modulators = bayesnewton.utils.softplus(posterior_var[:, num_subbands:])

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

prior_samples = np.squeeze(model.prior_sample(X=x, num_samps=1))
prior_samples_subbands = prior_samples[:, :num_subbands]
prior_samples_modulators = bayesnewton.utils.softplus(prior_samples[:, num_subbands:])
prior_samples_sig = np.sum(
    prior_samples_subbands * (model.likelihood.weights[None] @ prior_samples_modulators[..., None])[..., 0],
    axis=-1
)
plt.figure(3, figsize=(12, 5))
plt.clf()
plt.plot(x, prior_samples_sig, 'k', linewidth=0.6)
plt.xlim(x[0], x[-1])
plt.legend()
plt.xlabel('time (milliseconds)')

plt.figure(4, figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(x, prior_samples_subbands, linewidth=0.6)
plt.xlim(x[0], x[-1])
plt.title('subbands')
plt.subplot(2, 1, 2)
plt.plot(x, prior_samples_modulators, linewidth=0.6)
plt.xlim(x[0], x[-1])
plt.xlabel('time (milliseconds)')
plt.title('amplitude modulators')

plt.show()
