import bayesnewton
import objax
import numpy as np
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt
import pickle
import time

train_data = pickle.load(open(f'data/train_data_6_0.pickle', "rb"))
pred_data = pickle.load(open(f'data/pred_data_6_0.pickle', "rb"))
pred_data = pred_data['grid']

X = train_data['X']
Y = train_data['Y']
X_test = pred_data['X']
Y_test = pred_data['Y']

N = X.shape[0]  # number of data points
# print(N)

np.random.seed(123)
# print(Y.shape)
# print(X.shape)
# print(X)

# put test points on a grid to speed up prediction
t, r, Y = bayesnewton.utils.create_spatiotemporal_grid(X, Y)
t_test, r_test, Y_test = bayesnewton.utils.create_spatiotemporal_grid(X_test, Y_test)
print(t.shape)
print(r.shape)
print(Y.shape)
# print(t)
# print(r)
# print(Y)

# plt.imshow(Y, extent=(t[0, 0], t[-1, 0], r[0, 0], r[-1, 0]))
# plt.imshow(Y[..., 0].T)
# plt.show()

var_f = 1.  # GP variance
# var_f = 2.  # GP variance
len_f_time = 0.1  # temporal lengthscale
# len_f_time = 0.025  # temporal lengthscale
len_f_space = 0.1  # spatial lengthscale
var_y = 0.1  # observation noise variance
# var_y = 0.3  # observation noise variance

M = 6
# z = np.linspace(np.min(r) + 0.05, np.max(r) - 0.05, M)[:, None]  # inducing points
# z = kmeans2(r_test[0], M, minit="points")[0]
# print(z)
z = np.array([0.0862069, 0.96551724, 0.5, 0.29310345, 0.84482759, 0.68965517])
kern = bayesnewton.kernels.SpatioTemporalMatern32(variance=var_f, lengthscale_time=len_f_time, lengthscale_space=len_f_space,
                                           z=z, sparse=True, opt_z=True, conditional='Full')
lik = bayesnewton.likelihoods.Gaussian(variance=var_y)
model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=t, R=r, Y=Y)
# model = bayesnewton.models.InfiniteHorizonVariationalGP(kernel=kern, likelihood=lik, X=t, R=r, Y=Y)
# model = bayesnewton.models.MarkovVariationalGPMeanField(kernel=kern, likelihood=lik, X=t, R=r, Y=Y)

lr_adam = 0.01
lr_newton = 1.
iters = 250
opt_hypers = objax.optimizer.Adam(model.vars())
energy = objax.GradValues(model.energy, model.vars())


@objax.Function.with_vars(model.vars() + opt_hypers.vars())
def train_op():
    model.inference(lr=lr_newton)  # perform inference and update variational params
    dE, E = energy()  # compute energy and its gradients w.r.t. hypers
    opt_hypers(lr_adam, dE)
    return E


train_op = objax.Jit(train_op)

elbos = np.zeros(iters)
t0 = time.time()
for i in range(1, iters + 1):
    loss = train_op()
    print('iter %2d: energy: %1.4f' % (i, loss[0]))
    elbos[i-1] = loss[0]
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

print(model.likelihood.variance)
print(model.kernel.variance)
print(model.kernel.temporal_kernel.lengthscale)
print(model.kernel.spatial_kernel.lengthscale)

# calculate posterior predictive distribution via filtering and smoothing at train & test locations:
print('calculating the posterior predictive distribution ...')
t0 = time.time()
posterior_mean, posterior_var = model.predict(X=t_test, R=r_test)
nlpd = model.negative_log_predictive_density(X=t_test, R=r_test, Y=Y_test)
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))
print('nlpd: %2.3f' % nlpd)

print('plotting ...')
plt.figure(1)
im = plt.imshow(np.squeeze(Y).T, extent=[np.min(t), np.max(t), np.min(r), np.max(r)], origin='lower')
plt.colorbar(im, fraction=0.0235, pad=0.04)
plt.figure(2)
im = plt.imshow(np.squeeze(Y_test).T, extent=[np.min(t_test), np.max(t_test), np.min(r_test), np.max(r_test)], origin='lower')
plt.colorbar(im, fraction=0.0235, pad=0.04)
plt.figure(3)
im = plt.imshow(posterior_mean.T, extent=[np.min(t_test), np.max(t_test), np.min(r_test), np.max(r_test)], origin='lower')
plt.plot(np.min(t_test) * np.ones_like(model.kernel.z.value[:, 0]), model.kernel.z.value[:, 0], 'k>', markersize=6)
plt.colorbar(im, fraction=0.0235, pad=0.04)
plt.figure(4)
plt.plot(np.arange(iters), elbos)
plt.show()
