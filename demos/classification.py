import bayesnewton
import objax
import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(99)
N = 500  # number of training points
M = 20
Nbatch = 100
# x = 100 * np.random.rand(N)
x0 = 40 * np.random.rand(N//2)
x1 = 40 * np.random.rand(N//2) + 60
x = np.concatenate([x0, np.array([50]), x1], axis=0)
# x = np.linspace(np.min(x), np.max(x), N)
f = lambda x_: 6 * np.sin(np.pi * x_ / 10.0) / (np.pi * x_ / 10.0 + 1)
y_ = f(x) + np.sqrt(0.05)*np.random.randn(x.shape[0])
y = np.sign(y_)
y[y == -1] = 0
x_test = np.linspace(np.min(x)-5.0, np.max(x)+5.0, num=500)
y_test = np.sign(f(x_test) + np.sqrt(0.05)*np.random.randn(x_test.shape[0]))
y_test[y_test == -1] = 0
x_plot = np.linspace(np.min(x)-10.0, np.max(x)+10.0, num=500)
z = np.linspace(min(x), max(x), num=M)

x = x[:, None]
x_plot = x_plot[:, None]

var_f = 1.  # GP variance
len_f = 5.0  # GP lengthscale

kern = bayesnewton.kernels.Matern52(variance=var_f, lengthscale=len_f)
lik = bayesnewton.likelihoods.Bernoulli(link='logit')

# model = bayesnewton.models.VariationalGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.ExpectationPropagationGP(kernel=kern, likelihood=lik, X=x, Y=y, power=0.5)
# model = bayesnewton.models.SparseVariationalGP(kernel=kern, likelihood=lik, X=x, Y=y, Z=z)
model = bayesnewton.models.SparseExpectationPropagationGP(kernel=kern, likelihood=lik, X=x, Y=y, Z=z, power=0.5)
# model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.MarkovVariationalGaussNewtonGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.MarkovNewtonGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.MarkovQuasiNewtonGP(kernel=kern, likelihood=lik, X=x, Y=y, damped=True)
# model = bayesnewton.models.MarkovVariationalQuasiNewtonGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.MarkovExpectationPropagationQuasiNewtonGP(kernel=kern, likelihood=lik, X=x, Y=y, power=0.5)
# model = bayesnewton.models.MarkovExpectationPropagationRiemannGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.MarkovPosteriorLinearisationGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.MarkovPosteriorLinearisation2ndOrderGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.MarkovPosteriorLinearisation2ndOrderQuasiNewtonGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.MarkovTaylorNewtonGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.MarkovLaplaceGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.InfiniteHorizonVariationalGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.SparseMarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=x, Y=y, Z=z)

lr_adam = 0.05
lr_newton = 1
iters = 200
opt_hypers = objax.optimizer.Adam(model.vars())
energy = objax.GradValues(model.energy, model.vars())


@objax.Function.with_vars(model.vars() + opt_hypers.vars())
def train_op(ind):
    model.inference(batch_ind=ind, lr=lr_newton)  # perform inference and update variational params
    dE, E = energy(batch_ind=ind)  # compute energy and its gradients w.r.t. hypers
    opt_hypers(lr_adam, dE)
    return E


train_op = objax.Jit(train_op)

t0 = time.time()
for i in range(1, iters + 1):
    batch_ind = np.random.permutation(N)[:Nbatch]
    loss = train_op(batch_ind)
    print('iter %2d, energy: %1.4f' % (i, loss[0]))
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
plt.plot(x, y, 'b+', label='training observations')
plt.plot(x_test, y_test, 'r+', alpha=0.4, label='test observations')
plt.plot(x_plot, link_fn(posterior_mean), 'm', label='posterior mean')
plt.fill_between(x_plot[:, 0], link_fn(lb), link_fn(ub), color='m', alpha=0.05, label='95% confidence')
if hasattr(model, 'Z'):
    plt.plot(model.Z.value[:, 0], +0.03 * np.ones_like(model.Z.value[:, 0]), 'm^', markersize=5)
plt.xlim(x_plot[0], x_plot[-1])
plt.legend(loc=3)
plt.title('GP classification.')
plt.xlabel('$X$')
plt.show()
