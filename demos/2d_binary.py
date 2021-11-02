import bayesnewton
from bayesnewton.cubature import Unscented
import objax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

print('loading rainforest data ...')
data = np.loadtxt('../data/TRI2TU-data.csv', delimiter=',')

nr = 10  # spatial grid point (y-axis)
nt = 20  # temporal grid points (x-axis)
scale = 1000 / nt

t, r, Y_ = bayesnewton.utils.discretegrid(data, [0, 1000, 0, 500], [nt, nr])

np.random.seed(99)
N = nr * nt  # number of data points

# make binary for classification demo
Y_ = np.sign(Y_ - np.mean(Y_))
Y_[Y_ == -1] = 0

test_ind = np.random.permutation(N)[:N//4]
Y = Y_.flatten()
Y[test_ind] = np.nan
Y = Y.reshape(nt, nr)

# flatten for use in standard GP
X = np.vstack((t.flatten(), r.flatten())).T
Y_GP = Y.flatten()

var_f = 1.  # GP variance
len_f = 10.  # lengthscale

markov = True

lik = bayesnewton.likelihoods.Bernoulli()
if markov:
    kern = bayesnewton.kernels.SpatialMatern32(variance=var_f, lengthscale=len_f, z=r[0, ...], sparse=False)
    # flattened data version
    # kern = bayesnewton.kernels.SpatialMatern32(variance=var_f, lengthscale=len_f, z=r[0, ...], sparse=True, opt_z=False)

    model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=t, R=r, Y=Y)
    # model = bayesnewton.models.MarkovVariationalGPMeanField(kernel=kern, likelihood=lik, X=t, R=r, Y=Y)
else:
    kern = bayesnewton.kernels.SpatialMatern32(variance=var_f, lengthscale=len_f, z=r[0, ...], sparse=True, opt_z=False)
    model = bayesnewton.models.VariationalGP(kernel=kern, likelihood=lik, X=X, Y=Y_GP)

lr_adam = 0.2
lr_newton = 1.
iters = 20
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
if markov:
    posterior_mean, posterior_var = model.predict(X=t, R=r)
else:
    Xtest_GP = np.vstack((t.flatten(), r.flatten())).T
    posterior_mean, posterior_var = model.predict(X=Xtest_GP)
    posterior_mean = posterior_mean.reshape(nt, -1)
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))

link_fn = lik.link_fn

print('plotting ...')
cmap = cm.coolwarm
plt.figure(1, figsize=(10, 5))
plt.plot(data[:, 0], data[:, 1], 'k.', markersize=2)
plt.title('Tree locations')
plt.xlim(0, 1000)
plt.ylim(0, 500)
plt.figure(2, figsize=(10, 5))
im = plt.imshow(Y_.T, cmap=cmap, extent=[0, 1000, 0, 500], origin='lower')
cb = plt.colorbar(im, fraction=0.0235, pad=0.04)
cb.set_ticks([cb.vmin, 0, cb.vmax])
cb.set_ticklabels([0., 0.5, 1.])
plt.title('Tree count data (full).')
plt.figure(3, figsize=(10, 5))
im = plt.imshow(Y.T, cmap=cmap, extent=[0, 1000, 0, 500], origin='lower')
cb = plt.colorbar(im, fraction=0.0235, pad=0.04)
cb.set_ticks([cb.vmin, 0, cb.vmax])
cb.set_ticklabels([0., 0.5, 1.])
plt.title('Tree count data (with missing values).')
plt.figure(4, figsize=(10, 5))
im = plt.imshow(link_fn(posterior_mean).T, cmap=cmap, extent=[0, 1000, 0, 500], origin='lower')
cb = plt.colorbar(im, fraction=0.0235, pad=0.04)
cb.set_ticks([cb.vmin, 0.5, cb.vmax])
cb.set_ticklabels([0., 0.5, 1.])
plt.xlim(0, 1000)
plt.ylim(0, 500)
plt.title('2D classification (rainforest tree data). Tree intensity per $m^2$.')
plt.xlabel('first spatial dimension, $t$ (metres)')
plt.ylabel('second spatial dimension, $r$ (metres)')
plt.show()
