import sys
import bayesnewton
import objax
from bayesnewton.cubature import Unscented
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat
import pickle

# ----- THE BAYESNEWTON API HAS CHANGED SO THIS SCRIPT WILL NO LONGER RUN -----

if len(sys.argv) > 1:
    method = int(sys.argv[1])
    fold = int(sys.argv[2])
    plot_final = False
    save_result = True
    iters = 500
    M_ind = int(sys.argv[3])
    num_inducing = np.linspace(100, 2000, 20, dtype=int)
    M = num_inducing[M_ind]
else:
    method = 3
    fold = 0
    plot_final = True
    save_result = False
    iters = 50
    M_ind = 0
    num_inducing = np.linspace(100, 2000, 20, dtype=int)
    M = num_inducing[M_ind]

print('loading data ...')
y = loadmat('speech_female')['y']
fs = 44100  # sampling rate (Hz)
scale = 1000  # convert to milliseconds

normaliser = 0.5 * np.sqrt(np.var(y))
yTrain = y / normaliser  # rescale the data

N = y.shape[0]
x = np.linspace(0., N, num=N) / fs * scale  # arbitrary evenly spaced inputs inputs
batch_size = 20000
# z = np.linspace(x[0], x[-1], num=M)
z_all = np.linspace(x[0], x[-1], 2000)
np.random.seed(99)
z_ind = np.random.permutation(2000)
z = z_all[np.sort(z_ind[:M])]

np.random.seed(123)
# 10-fold cross-validation setup
ind_shuffled = np.random.permutation(N)
ind_split = np.stack(np.split(ind_shuffled, 10))  # 10 random batches of data indices

print('method number', method)
print('batch number', fold)
print('num inducing', z.shape[0])

# Get training and test indices
ind_test = ind_split[fold]  # np.sort(ind_shuffled[:N//10])
ind_train = np.concatenate(ind_split[np.arange(10) != fold])
x_train = x[ind_train]  # 90/10 train/test split
x_test = x[ind_test]
y_train = y[ind_train]
y_test = y[ind_test]

fundamental_freq = 220  # Hz
radial_freq = 2 * np.pi * fundamental_freq / scale  # radial freq = 2pi * f / scale
sub1 = bayesnewton.kernels.SubbandMatern12(variance=.1, lengthscale=75., radial_frequency=radial_freq, fix_variance=True)
# 1st harmonic
sub2 = bayesnewton.kernels.SubbandMatern12(variance=.1, lengthscale=75., radial_frequency=2 * radial_freq, fix_variance=True)
# 2nd harmonic
sub3 = bayesnewton.kernels.SubbandMatern12(variance=.1, lengthscale=75., radial_frequency=3 * radial_freq, fix_variance=True)
mod1 = bayesnewton.kernels.Matern52(variance=.5, lengthscale=10., fix_variance=True)
mod2 = bayesnewton.kernels.Matern52(variance=.5, lengthscale=10., fix_variance=True)
mod3 = bayesnewton.kernels.Matern52(variance=.5, lengthscale=10., fix_variance=True)

kern = bayesnewton.kernels.Independent([sub1, sub2, sub3, mod1, mod2, mod3])

lik = bayesnewton.likelihoods.AudioAmplitudeDemodulation(variance=0.3)

if method == 0:
    inf = bayesnewton.inference.Taylor(
        cubature=Unscented(),
        energy_function=bayesnewton.inference.ExpectationPropagation(power=0.5, cubature=Unscented()).energy
    )
elif method == 1:
    inf = bayesnewton.inference.PosteriorLinearisation(
        cubature=Unscented(),
        energy_function=bayesnewton.inference.ExpectationPropagation(power=0.5, cubature=Unscented()).energy
    )
elif method == 2:
    inf = bayesnewton.inference.ExpectationPropagation(power=0.9, cubature=Unscented())  # power=1 unstable in sparse case
elif method == 3:
    inf = bayesnewton.inference.ExpectationPropagation(power=0.5, cubature=Unscented())
elif method == 4:
    inf = bayesnewton.inference.ExpectationPropagation(power=0.01, cubature=Unscented())
elif method == 5:
    inf = bayesnewton.inference.VariationalInference(cubature=Unscented())

model = bayesnewton.basemodels.SparseMarkovGaussianProcess(kernel=kern, likelihood=lik, X=x_train, Y=y_train, Z=z)

trainable_vars = model.vars() + inf.vars()
energy = objax.GradValues(inf.energy, trainable_vars)

lr_adam = 0.05
lr_newton = 0.05
opt = objax.optimizer.Adam(trainable_vars)


def train_op():
    inf(model, lr=lr_newton)  # perform inference and update variational params
    dE, E = energy(model)  # compute energy and its gradients w.r.t. hypers
    opt(lr_adam, dE)
    return E


train_op = objax.Jit(train_op, trainable_vars + opt.vars())


t0 = time.time()
for i in range(1, iters + 1):
    loss = train_op()
    print('iter %2d, energy: %1.4f' % (i, loss[0]))
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

# calculate posterior predictive distribution via filtering and smoothing at train & test locations:
print('calculating the posterior predictive distribution ...')
t0 = time.time()
predict_mean, predict_var = model.predict(X=x_test)
nlpd = model.negative_log_predictive_density(X=x_test, Y=y_test)

predict_mean_subbands = predict_mean[:, :3]
predict_mean_modulators = bayesnewton.utils.softplus(predict_mean[:, 3:])
predict_mean_sig = np.sum(predict_mean_subbands * predict_mean_modulators, axis=-1)

rmse = np.sqrt(np.mean((np.squeeze(predict_mean_sig) - np.squeeze(y_test)) ** 2))
nlml = loss[0]
t1 = time.time()
print('NLML: %1.2f' % nlml)
print('NLPD: %1.2f' % nlpd)
print('RMSE: %1.2f' % rmse)
print('prediction time: %2.2f secs' % (t1-t0))

if save_result:
    # with open("output/" + str(method) + "_" + str(fold) + "_nlpd.txt", "wb") as fp:
    #     pickle.dump(nlpd, fp)
    with open("output/varyM" + str(method) + "_" + str(fold) + ".txt", "rb") as fp:
        results_data = pickle.load(fp)
    results_data[M_ind, 0] = nlml
    results_data[M_ind, 1] = nlpd
    results_data[M_ind, 2] = rmse
    with open("output/varyM" + str(method) + "_" + str(fold) + ".txt", "wb") as fp:
        pickle.dump(results_data, fp)
    # with open("output/" + str(method) + "_" + str(fold) + "_nlpd.txt", "rb") as fp:
    #     nlpd_show = pickle.load(fp)
    # print(nlpd_show)

if plot_final:
    posterior_mean, posterior_var = model.predict(X=x)
    # lb = posterior_mean[:, 0] - np.sqrt(posterior_var[:, 0]) * 1.96
    # ub = posterior_mean[:, 0] + np.sqrt(posterior_var[:, 0]) * 1.96

    posterior_mean_subbands = posterior_mean[:, :3]
    posterior_mean_modulators = bayesnewton.utils.softplus(posterior_mean[:, 3:])
    posterior_mean_sig = np.sum(posterior_mean_subbands * posterior_mean_modulators, axis=-1)
    posterior_var_subbands = posterior_var[:, :3]
    posterior_var_modulators = bayesnewton.utils.softplus(posterior_var[:, 3:])

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
