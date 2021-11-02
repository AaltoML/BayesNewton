import bayesnewton
import objax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import tikzplotlib

print('loading rainforest data ...')
data = np.loadtxt('../data/TRI2TU-data.csv', delimiter=',')

nr = 50  # spatial grid point (y-axis)
nt = 100  # temporal grid points (x-axis)
binsize = 1000 / nt

t, r, Y_ = bayesnewton.utils.discretegrid(data, [0, 1000, 0, 500], [nt, nr])
t_flat, r_flat, Y_flat = t.flatten(), r.flatten(), Y_.flatten()

N = nr * nt  # number of data points

np.random.seed(99)
test_ind = np.random.permutation(N)[:N//10]
t_test = t_flat[test_ind]
r_test = r_flat[test_ind]
Y_test = Y_flat[test_ind]
Y_flat[test_ind] = np.nan
Y = Y_flat.reshape(nt, nr)

# put test points on a grid to speed up prediction
X_test = np.concatenate([t_test[:, None], r_test[:, None]], axis=1)
t_test, r_test, Y_test = bayesnewton.utils.create_spatiotemporal_grid(X_test, Y_test)

var_f = 1.  # GP variance
len_f = 20.  # lengthscale

kern = bayesnewton.kernels.SpatialMatern32(variance=var_f, lengthscale=len_f, z=r[0, ...], sparse=False)
# kern = bayesnewton.kernels.SpatialMatern32(variance=var_f, lengthscale=len_f, z=r[0, ...], sparse=True)
lik = bayesnewton.likelihoods.Poisson(binsize=binsize)
# lik = bayesnewton.likelihoods.Gaussian(variance=1)
# model = bayesnewton.models.VariationalGP(kernel=kern, likelihood=lik, X=x, Y=Y)
model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=t, R=r, Y=Y)
# model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=t_flat, R=r_flat, Y=Y_flat)
# model = bayesnewton.models.InfiniteHorizonVariationalGP(kernel=kern, likelihood=lik, X=t, R=r, Y=Y)
# model = bayesnewton.models.MarkovVariationalGPMeanField(kernel=kern, likelihood=lik, X=t, R=r, Y=Y)

lr_adam = 0.2
lr_newton = 0.2
iters = 10
opt_hypers = objax.optimizer.Adam(model.vars())
energy = objax.GradValues(model.energy, model.vars())


@objax.Function.with_vars(model.vars() + opt_hypers.vars())
def train_op():
    model.inference(lr=lr_newton)  # perform inference and update variational params
    dE, E = energy()  # compute energy and its gradients w.r.t. hypers
    opt_hypers(lr_adam, dE)
    test_nlpd_ = model.negative_log_predictive_density(X=t_test, R=r_test, Y=Y_test)
    return E, test_nlpd_


train_op = objax.Jit(train_op)

t0 = time.time()
for i in range(1, iters + 1):
    loss, test_nlpd = train_op()
    print('iter %2d, energy: %1.4f, nlpd: %1.4f' % (i, loss[0], test_nlpd))
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

# calculate posterior predictive distribution via filtering and smoothing at train & test locations:
print('calculating the posterior predictive distribution ...')
t0 = time.time()
posterior_mean, posterior_var = model.predict(X=t, R=r)
# posterior_mean_y, posterior_var_y = model.predict_y(X=t, R=r)
nlpd = model.negative_log_predictive_density(X=t_test, R=r_test, Y=Y_test)
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))
print('nlpd: %2.3f' % nlpd)

link_fn = lik.link_fn

print('plotting ...')
cmap = cm.viridis
plt.figure(1, figsize=(10, 5))
plt.plot(data[:, 0], data[:, 1], 'k.', markersize=2)
plt.title('Tree locations')
plt.xlim(0, 1000)
plt.ylim(0, 500)
plt.figure(2, figsize=(10, 5))
im = plt.imshow(Y_.T / binsize, cmap=cmap, extent=[0, 1000, 0, 500], origin='lower')
plt.colorbar(im, fraction=0.0235, pad=0.04)
plt.title('Tree count data (full).')
plt.figure(3, figsize=(10, 5))
im = plt.imshow(Y.T / binsize, cmap=cmap, extent=[0, 1000, 0, 500], origin='lower')
plt.colorbar(im, fraction=0.0235, pad=0.04)
plt.title('Tree count data (with missing values).')
plt.figure(4, figsize=(10, 5))
im = plt.imshow(link_fn(posterior_mean).T, cmap=cmap, extent=[0, 1000, 0, 500], origin='lower')
# im = plt.imshow(posterior_mean_y.T, cmap=cmap, extent=[0, 1000, 0, 500], origin='lower')
plt.colorbar(im, fraction=0.0235, pad=0.04)
plt.xlim(0, 1000)
plt.ylim(0, 500)
# plt.title('2D log-Gaussian Cox process (rainforest tree data). Log-intensity shown.')
plt.title('2D log-Gaussian Cox process (rainforest tree data). Tree intensity per $m^2$.')
plt.xlabel('first spatial dimension, $t$ (metres)')
plt.ylabel('second spatial dimension, $r$ (metres)')


# plt.figure(5, figsize=(10, 5))
# plt.plot(data[:, 0], data[:, 1], 'k.', markersize=2)
# bayesnewton.utils.bitmappify(plt.gca(), 200)
# plt.xlabel('first spatial dimension, $t$ (metres)')
# plt.ylabel('second spatial dimension, $\\Space$ (metres)')
# plt.xlim(0, 1000)
# plt.ylim(0, 500)
# tikzplotlib.save('/Users/wilkinw1/postdoc/inprogress/ati-fcai/paper/icml2021/fig/tree_locations.tex',
#                  axis_width='\\figurewidth',
#                  axis_height='\\figureheight',
#                  tex_relative_path_to_data='./fig/')
#
# plt.figure(6, figsize=(10, 5))
# im = plt.imshow(link_fn(posterior_mean).T, cmap=cmap, extent=[0, 1000, 0, 500], origin='lower')
# plt.xlim(0, 1000)
# plt.ylim(0, 500)
# plt.xlabel('first spatial dimension, $t$ (metres)')
# plt.ylabel('\\phantom{second spatial dimension, $\\Space$ (metres)}')
# tikzplotlib.save('/Users/wilkinw1/postdoc/inprogress/ati-fcai/paper/icml2021/fig/tree_posterior.tex',
#                  axis_width='\\figurewidth',
#                  axis_height='\\figureheight',
#                  tex_relative_path_to_data='./fig/')

plt.show()
