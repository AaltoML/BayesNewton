import sys
import bayesnewton
import objax
from bayesnewton.cubature import Unscented
import numpy as np
import time
from scipy.io import loadmat
from jax.lib import xla_bridge
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
    # fold = int(sys.argv[2])
    # plot_final = False
    # save_result = True
else:
    method = 2
    # fold = 0
    # plot_final = True
    # save_result = False

# if len(sys.argv) > 2:
#     baseline = bool(int(sys.argv[2]))
# else:
#     baseline = True
baseline = True

if len(sys.argv) > 2:
    parallel = bool(int(sys.argv[2]))
else:
    parallel = None

if len(sys.argv) > 3:
    num_components = int(sys.argv[3])
else:
    num_components = 3

time_steps = [5000, 10000, 15000, 20000]
if len(sys.argv) > 4:
    num_time_steps_ind = int(sys.argv[4])
else:
    num_time_steps_ind = 3

num_time_steps = time_steps[num_time_steps_ind]

# if len(sys.argv) > 6:
#     iters = int(sys.argv[6])
# else:
iters = 11

print('method number:', method)
# print('batch number:', fold)
# print('baseline:', baseline)
print('parallel:', parallel)
print('num components:', num_components)
print('num time steps:', num_time_steps)
# print('num iterations:', iters)

x_train = x[:num_time_steps]
y_train = y[:num_time_steps]

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


if method == 0:
    inf = bayesnewton.inference.Taylor
elif method == 1:
    inf = bayesnewton.inference.PosteriorLinearisation
elif method in [2, 3, 4]:
    inf = bayesnewton.inference.ExpectationPropagation
elif method == 5:
    inf = bayesnewton.inference.VariationalInference

if baseline:
    mod = bayesnewton.basemodels.MarkovGP
    Mod = bayesnewton.build_model(mod, inf)
    model = Mod(kernel=kern, likelihood=lik, X=x_train, Y=y_train, parallel=parallel)
else:
    mod = bayesnewton.basemodels.SparseMarkovGaussianProcess
    Mod = bayesnewton.build_model(mod, inf)
    model = Mod(kernel=kern, likelihood=lik, X=x_train, Y=y_train, Z=z, parallel=parallel)

unscented_transform = Unscented(dim=num_components)  # 5th-order unscented transform

if method == 2:
    inf_args = {"power": 1., "cubature": unscented_transform}
elif method == 3:
    inf_args = {"power": 0.5, "cubature": unscented_transform}
elif method == 4:
    inf_args = {"power": 0.01, "cubature": unscented_transform}
else:
    inf_args = {"cubature": unscented_transform}


lr_adam = 0.05
lr_newton = 0.05
opt = objax.optimizer.Adam(model.vars())
energy = objax.GradValues(model.energy, model.vars())


@objax.Function.with_vars(model.vars() + opt.vars())
def train_op():
    model.inference(lr=lr_newton, **inf_args)  # perform inference and update variational params
    dE, E = energy(**inf_args)  # compute energy and its gradients w.r.t. hypers
    opt(lr_adam, dE)
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
avg_time_taken = (t3-t2)/(iters-1)
print('per-iteration time (excl. compile): %2.2f secs' % avg_time_taken)

cpugpu = xla_bridge.get_backend().platform

with open("output/" + str(method) + "_" + str(num_time_steps_ind) + "_" + str(num_components) + "_"
          + str(int(parallel)) + "_" + cpugpu + ".txt", "wb") as fp:
    pickle.dump(avg_time_taken, fp)
