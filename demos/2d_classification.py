import bayesnewton
import objax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

# --- small data set ---
# inputs = np.loadtxt('../data/banana_X_train', delimiter=',')
# X = inputs[:, :1]  # temporal inputs (x-axis)
# R = inputs[:, 1:]  # spatial inputs (y-axis)
# Y = np.loadtxt('../data/banana_Y_train')[:, None]  # observations / labels

# --- large data set ---
inputs = np.loadtxt('../data/banana_large.csv', delimiter=',', skiprows=1)
X = inputs[:, :1]  # temporal inputs (x-axis)
R = inputs[:, 1:2]  # spatial inputs (y-axis)
Y = np.maximum(inputs[:, 2:], 0)  # observations / labels

# Test points
Xtest, Rtest = np.mgrid[-2.8:2.8:100j, -2.8:2.8:100j]
Xtest_GP = np.vstack((Xtest.flatten(), Rtest.flatten())).T
# X0test, X1test = np.linspace(-3., 3., num=100), np.linspace(-3., 3., num=100)

Mt = 15  # num inducing points in time
Ms = 15  # num inducing points in space
batch_size = X.shape[0]
Z = np.linspace(-3., 3., Mt)[:, None]  # inducing points

np.random.seed(99)
N = X.shape[0]  # number of training points

var_f = 0.3  # GP variance
len_time = 0.3  # temporal lengthscale
len_space = 0.3  # spacial lengthscale

markov = True

kern = bayesnewton.kernels.SpatioTemporalMatern52(variance=var_f,
                                                  lengthscale_time=len_time,
                                                  lengthscale_space=len_space,
                                                  z=np.linspace(-3, 3, Ms),
                                                  sparse=True,
                                                  opt_z=True,
                                                  conditional='Full')
lik = bayesnewton.likelihoods.Bernoulli(link='logit')
if markov:
    # model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=X, R=R, Y=Y)
    # model = bayesnewton.models.MarkovVariationalGPMeanField(kernel=kern, likelihood=lik, X=X, R=R, Y=Y)
    model = bayesnewton.models.SparseMarkovVariationalGP(kernel=kern, likelihood=lik, X=X, R=R, Y=Y, Z=Z)
    # model = bayesnewton.models.SparseMarkovVariationalGPMeanField(kernel=kern, likelihood=lik, X=X, R=R, Y=Y, Z=Z)
    # model = bayesnewton.models.SparseInfiniteHorizonVariationalGP(kernel=kern, likelihood=lik, X=X, R=R, Y=Y, Z=Z)
else:
    model = bayesnewton.models.VariationalGP(kernel=kern, likelihood=lik, X=inputs, Y=Y)  # TODO: this model is not sparse

lr_adam = 0.1
lr_newton = 0.5
iters = 25
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

# calculate posterior predictive distribution via filtering and smoothing at train & test locations:
print('calculating the posterior predictive distribution ...')
t0 = time.time()
if markov:
    posterior_mean, posterior_var = model.predict(X=Xtest, R=Rtest)
else:
    posterior_mean, posterior_var = model.predict(X=Xtest_GP)
    Ntest = Xtest.shape[0]
    posterior_mean = posterior_mean.reshape(Ntest, -1)
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))

link_fn = lik.link_fn

print('plotting ...')
z_final = model.kernel.z.value.reshape(-1, 1)
cmap = cm.coolwarm
ax, fig = plt.subplots(1, figsize=(6, 6))
for label, mark in [[1, 'o'], [0, 'o']]:
    ind = Y[:, 0] == label
    plt.scatter(X[ind], R[ind], color=cmap(label - 0.01), s=50, alpha=.5, edgecolor='k')
plt.contour(Xtest, Rtest, posterior_mean, levels=[.0], colors='k', linewidths=4.)
# plt.axis('equal')
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
# ax.axis('off')
ax = plt.gca()
ax.axis('equal')
ax.axis('square')
lim = 2.8
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
if markov:
    # plt.plot(Xtest[:, 0], np.tile(z_final, [1, Xtest.shape[0]]).T, '--', color='k', linewidth=1.)
    plt.plot((-lim+0.05) * np.ones_like(z_final), z_final, ">", color='k', markersize=6)
    if hasattr(model, 'Z'):
        plt.plot(model.Z.value[:, 0], 0.06-lim * np.ones_like(model.Z.value[:, 0]), 'k^', markersize=6)

ax2, fig2 = plt.subplots(1, figsize=(6, 6))
im = plt.imshow(link_fn(posterior_mean).T, cmap=cmap, extent=[-lim, lim, -lim, lim], origin='lower')
cb = plt.colorbar(im, fraction=0.046, pad=0.04)
cb.set_ticks([cb.vmin, 0, cb.vmax])
cb.set_ticklabels([-1, 0, 1])
# plt.contour(Xtest, Rtest, mu, levels=[.0], colors='k', linewidths=1.5)
# plt.axis('equal')
for label in [1, 0]:
    ind = Y[:, 0] == label
    plt.scatter(X[ind], R[ind], color=cmap(label - 0.01), s=50, alpha=.25, edgecolor='k')
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
ax2 = plt.gca()
ax2.axis('equal')
ax2.axis('square')
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
if markov:
    # plt.plot(Xtest[:, 0], np.tile(z_final, [1, Xtest.shape[0]]).T, '--', color='w', linewidth=1.)
    plt.plot((-lim+0.05) * np.ones_like(z_final), model.kernel.z.value, ">", color='w', markersize=6)
    if hasattr(model, 'Z'):
        plt.plot(model.Z.value[:, 0], 0.06-lim * np.ones_like(model.Z.value[:, 0]), 'w^', markersize=6)
plt.show()
