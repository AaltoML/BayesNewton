import newt
import objax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

print('loading coal data ...')
disaster_timings = pd.read_csv('../data/coal.txt', header=None).values[:, 0]

# Discretization
num_time_bins = 200
# Discretize the data
x = np.linspace(min(disaster_timings), max(disaster_timings), num_time_bins).T
y = np.histogram(disaster_timings, np.concatenate([[-1e10], x[:-1] + np.diff(x)/2, [1e10]]))[0][:, None]
# Test points
x_test = x
x_plot = np.linspace(np.min(x_test)-5, np.max(x_test)+5, 200)
M = 15
z = np.linspace(np.min(x), np.max(x), M)

x = x[:, None]

meanval = np.log(len(disaster_timings)/num_time_bins)  # TODO: incorporate mean
binsize = (max(x) - min(x)) / num_time_bins

var_f = 1.0  # GP variance
len_f = 4.  # GP lengthscale

kern = newt.kernels.Matern52(variance=var_f, lengthscale=len_f)
lik = newt.likelihoods.Poisson(binsize=binsize)
# model = newt.models.GP(kernel=kern, likelihood=lik, X=x, Y=y)
model = newt.models.MarkovGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = newt.models.SparseMarkovGP(kernel=kern, likelihood=lik, X=x, Y=y, Z=z)

# inf = newt.inference.VariationalInference()
inf = newt.inference.ExpectationPropagation(power=0.01)

trainable_vars = model.vars() + inf.vars()
energy = objax.GradValues(inf.energy, trainable_vars)

lr_adam = 0.1
lr_newton = 1
iters = 100
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
print('calculating the posterior predictive distribution ...')
t0 = time.time()
posterior_mean, posterior_var = model.predict(X=x_plot)
# posterior_mean_y, posterior_var_y = model.predict_y(X=x_plot)
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))

link_fn = lik.link_fn

post_mean_lgcp = link_fn(posterior_mean + posterior_var / 2)
lb_lgcp = link_fn(posterior_mean - np.sqrt(posterior_var) * 1.645)
ub_lgcp = link_fn(posterior_mean + np.sqrt(posterior_var) * 1.645)

# lb_y = posterior_mean_y - 1.96 * np.sqrt(posterior_var_y)
# ub_y = posterior_mean_y + 1.96 * np.sqrt(posterior_var_y)

print('plotting ...')
plt.figure(1, figsize=(12, 5))
plt.clf()
plt.plot(disaster_timings, 0*disaster_timings, 'k+', label='observations', clip_on=False)
plt.plot(x_plot, post_mean_lgcp, 'g', label='posterior mean')
# plt.plot(x_plot, posterior_mean_y, 'r', label='posterior mean (y)')
plt.fill_between(x_plot, lb_lgcp, ub_lgcp, color='g', alpha=0.05, label='95% confidence')
# plt.fill_between(x_plot, lb_y, ub_y, color='r', alpha=0.05, label='95% confidence (y)')
plt.xlim(x_plot[0], x_plot[-1])
plt.ylim(0.0)
plt.legend()
plt.title('log-Gaussian Cox process via Kalman smoothing (coal mining disasters)')
plt.xlabel('year')
plt.ylabel('accident intensity')
plt.show()
