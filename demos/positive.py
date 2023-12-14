import bayesnewton
import objax
import numpy as np
import matplotlib.pyplot as plt
import time


def nonlinearity(f_):
    return bayesnewton.utils.softplus(f_)


def wiggly_time_series(x_):
    return 2 * np.cos(0.06*x_+0.33*np.pi) * np.sin(0.4*x_) - 1.


np.random.seed(99)
N = 500  # number of training points
x = 100 * np.random.rand(N)
# x = np.linspace(np.min(x), np.max(x), N)
# f = lambda x_: 3 * np.sin(np.pi * x_ / 10.0)
f = wiggly_time_series
y = nonlinearity(f(x)) + np.sqrt(0.1)*np.random.randn(x.shape[0])
x_test = np.linspace(np.min(x), np.max(x), num=500)
y_test = nonlinearity(f(x_test)) + np.sqrt(0.05)*np.random.randn(x_test.shape[0])
x_plot = np.linspace(np.min(x)-10.0, np.max(x)+10.0, num=500)

M = 20
Z = np.linspace(np.min(x), np.max(x), M)

x = x[:, None]
x_plot = x_plot[:, None]

var_f = 1.  # GP variance
len_f = 5.0  # GP lengthscale
var_y = 5.0  # likelihood lengthscale

kern = bayesnewton.kernels.Matern52(variance=var_f, lengthscale=len_f)
lik = bayesnewton.likelihoods.Positive(variance=var_y)

# model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=x, Y=y, power=0.5)
model = bayesnewton.models.SparseVariationalGP(kernel=kern, likelihood=lik, X=x, Y=y, Z=Z, opt_z=True)
# model = bayesnewton.models.MarkovVariationalGaussNewtonGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.MarkovVariationalQuasiNewtonGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.MarkovExpectationPropagationQuasiNewtonGP(kernel=kern, likelihood=lik, X=x, Y=y, power=0.5)
# model = bayesnewton.models.SparseVariationalQuasiNewtonGP(kernel=kern, likelihood=lik, X=x, Y=y, Z=Z, opt_z=True)

lr_adam = 0.1
lr_newton = 0.3
iters = 1000
opt_hypers = objax.optimizer.Adam(model.vars())
energy = objax.GradValues(model.energy, model.vars())

damping = 0.5


@objax.Function.with_vars(model.vars() + opt_hypers.vars())
def train_op():
    model.inference(lr=lr_newton, damping=damping)  # perform inference and update variational params
    dE, E = energy()  # compute energy and its gradients w.r.t. hypers
    opt_hypers(lr_adam, dE)
    test_nlpd_ = model.negative_log_predictive_density(X=x_test, Y=y_test)
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
posterior_mean, posterior_var = model.predict(X=x_plot)
nlpd = model.negative_log_predictive_density(X=x_test, Y=y_test)
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))
print('nlpd: %2.3f' % nlpd)
lb = np.squeeze(posterior_mean) - 1.96 * np.squeeze(posterior_var) ** 0.5
ub = np.squeeze(posterior_mean) + 1.96 * np.squeeze(posterior_var) ** 0.5
link_fn = lik.link_fn

print('plotting ...')
plt.figure(1, figsize=(12, 5))
plt.clf()
plt.plot(x, y, 'b.', label='training observations')
plt.plot(x_test, y_test, 'r.', alpha=0.4, label='test observations')
plt.plot(x_plot, link_fn(np.squeeze(posterior_mean)), 'm', label='posterior mean')
plt.fill_between(x_plot[:, 0], link_fn(lb), link_fn(ub), color='m', alpha=0.05, label='95% confidence')
plt.xlim(x_plot[0], x_plot[-1])
if hasattr(model, 'Z'):
    plt.plot(model.Z.value[:, 0],
             (np.min(link_fn(lb))-1.)*np.ones_like(model.Z.value[:, 0]),
             'm^',
             markersize=4)
plt.legend(loc=3)
plt.xlabel('$X$')
plt.show()
