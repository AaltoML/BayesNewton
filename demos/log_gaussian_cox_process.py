import bayesnewton
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

kern = bayesnewton.kernels.Matern52(variance=var_f, lengthscale=len_f)
lik = bayesnewton.likelihoods.Poisson(binsize=binsize, link='logistic')

# model = bayesnewton.models.VariationalGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.SparseMarkovVariationalGP(kernel=kern, likelihood=lik, X=x, Y=y, Z=z)
# model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.MarkovVariationalGaussNewtonGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.MarkovVariationalQuasiNewtonGP(kernel=kern, likelihood=lik, X=x, Y=y)
model = bayesnewton.models.MarkovExpectationPropagationQuasiNewtonGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.MarkovGaussNewtonGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.MarkovTaylorGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.MarkovTaylorNewtonGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.MarkovLaplaceGP(kernel=kern, likelihood=lik, X=x, Y=y)

lr_adam = 0.1
lr_newton = 1
iters = 100
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
posterior_mean, posterior_var = model.predict(X=x_plot)
# posterior_mean_y, posterior_var_y = model.predict_y(X=x_plot)
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))

link_fn = lik.link_fn

post_mean_lgcp = link_fn(posterior_mean + posterior_var / 2)
lb_lgcp = link_fn(posterior_mean - np.sqrt(posterior_var) * 1.645)
ub_lgcp = link_fn(posterior_mean + np.sqrt(posterior_var) * 1.645)

# lb_y = posterior_mean_y - np.sqrt(posterior_var_y)
# ub_y = posterior_mean_y + np.sqrt(posterior_var_y)

print('plotting ...')
plt.figure(1, figsize=(12, 5))
plt.clf()
plt.plot(disaster_timings, 0*disaster_timings, 'k+', label='observations', clip_on=False)
plt.plot(x_plot, post_mean_lgcp, 'g', label='posterior mean')
# plt.plot(x_plot, posterior_mean_y, 'r', label='posterior mean (y)')
plt.fill_between(x_plot, lb_lgcp, ub_lgcp, color='g', alpha=0.05, label='95% confidence')
# plt.fill_between(x_plot, lb_y, ub_y, color='r', alpha=0.05, label='1 std (y)')
plt.xlim(x_plot[0], x_plot[-1])
plt.ylim(0.0)
plt.legend()
plt.title('log-Gaussian Cox process via Kalman smoothing (coal mining disasters)')
plt.xlabel('year')
plt.ylabel('accident intensity')
plt.show()
