import bayesnewton
import objax
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat

plot_final = True
plot_intermediate = False


print('loading data ...')
N_sig = 22050
# y = loadmat('../experiments/audio/speech_female')['y'][1000:N_sig+1000]
y = loadmat('../experiments/audio/speech_female')['y'][:N_sig]
fs = 44100  # sampling rate (Hz)
scale = 1000  # convert to milliseconds

normaliser = 0.5 * np.sqrt(np.var(y))
yTrain = y / normaliser  # rescale the data

# N = y.shape[0]
# yTrain = yTrain[:N]
x = np.linspace(0., N_sig, num=N_sig) / fs * scale  # arbitrary evenly spaced inputs inputs

np.random.seed(123)

gap_size = 2000
gap = np.arange(gap_size)
gap0 = gap + 3000
gap1 = gap + 8000
gap2 = gap + 13000
gap3 = gap + 18000
gaps = np.concatenate([gap0, gap1, gap2, gap3])
mask = np.ones_like(x, dtype=bool)
mask[gaps] = False

# x_train = x[:N]
x_train = x[mask]
x_test = x
# y_train = y[:N]
y_train = y[mask]
y_test = y

# N = 5000
N = x_train.shape[0]
batch_size = N
M = 30
z = np.linspace(x[0], x[-1], M)

var_f = 1.0  # GP variance
len_per = 1.0  # GP lengthscale
len_mat = 50.0
var_y = 0.1  # observation noise
fundamental_freq = 220  # Hz
radial_freq = 2 * np.pi * fundamental_freq / scale  # radial freq = 2pi * f / scale
per = 6.

# kern = bayesnewton.kernels.Matern72(variance=var_f, lengthscale=len_f)
kern = bayesnewton.kernels.QuasiPeriodicMatern32(variance=var_f, lengthscale_periodic=len_per, period=per, lengthscale_matern=len_mat)
# kern = bayesnewton.kernels.SubbandMatern12(variance=var_f, lengthscale=len_per, radial_frequency=radial_freq)
lik = bayesnewton.likelihoods.Gaussian(variance=var_y)

# kern = bayesnewton.kernels.Independent([sub1, sub2, sub3, mod1, mod2, mod3])
# lik = bayesnewton.likelihoods.AudioAmplitudeDemodulation(variance=0.3)

model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train)
# model = bayesnewton.models.SparseMarkovVariationalGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train, Z=z)


lr_adam = 0.1
lr_newton = 1.
iters = 100
opt_hypers = objax.optimizer.Adam(model.vars())
energy = objax.GradValues(model.energy, model.vars())


@objax.Function.with_vars(model.vars() + opt_hypers.vars())
def train_op():
    batch = np.random.permutation(N)[:batch_size]
    model.inference(lr=lr_newton, batch_ind=batch)  # perform inference and update variational params
    dE, E = energy(batch_ind=batch)  # compute energy and its gradients w.r.t. hypers
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
posterior_mean, posterior_var = model.predict(X=x_test)
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))

lb = posterior_mean - 1.96 * posterior_var ** 0.5
ub = posterior_mean + 1.96 * posterior_var ** 0.5

print('plotting ...')
plt.figure(1, figsize=(13, 7))
plt.clf()
plt.plot(x_test, y_test, 'r.', alpha=0.4, label='test observations', markersize=5)
plt.plot(x_train, y_train, 'k.', label='training observations', markersize=5)
# plt.plot(x_test[N:], y_test[N:], 'r.', alpha=0.4, label='test observations', markersize=5)
plt.plot(x_test, posterior_mean, 'b-', label='posterior mean')
plt.fill_between(x_test[..., 0], lb, ub, color='b', alpha=0.05, label='95% confidence')
if hasattr(model, 'Z'):
    plt.plot(model.Z[:, 0], (np.min(lb)-0.1)*np.ones_like(model.Z[:, 0]), 'r^', markersize=4)
plt.xlim([x_test[0], x_test[-1]])
# plt.ylim([-2, 2])
plt.legend(loc=0)
plt.title('Sparse GP regression via Kalman smoothing.')
plt.xlabel('time (milliseconds)')
plt.show()
