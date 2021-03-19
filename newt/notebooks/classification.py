import newt
import objax
import numpy as np
import matplotlib.pyplot as plt
import time

print('generating some data ...')
np.random.seed(99)
N = 500  # number of training points
M = 20
# x = 100 * np.random.rand(N)
x0 = 40 * np.random.rand(N//2)
x1 = 40 * np.random.rand(N//2) + 60
x = np.concatenate([x0, np.array([50]), x1], axis=0)
# x = np.linspace(np.min(x), np.max(x), N)
f = lambda x_: 6 * np.sin(np.pi * x_ / 10.0) / (np.pi * x_ / 10.0 + 1)
y_ = f(x) + np.math.sqrt(0.05)*np.random.randn(x.shape[0])
y = np.sign(y_)
y[y == -1] = 0
x_test = np.linspace(np.min(x)-5.0, np.max(x)+5.0, num=500)
y_test = np.sign(f(x_test) + np.math.sqrt(0.05)*np.random.randn(x_test.shape[0]))
y_test[y_test == -1] = 0
x_plot = np.linspace(np.min(x)-10.0, np.max(x)+10.0, num=500)
z = np.linspace(min(x), max(x), num=M)

x = x[:, None]
x_plot = x_plot[:, None]

var_f = 1.  # GP variance
len_f = 5.0  # GP lengthscale

kern = newt.kernels.Matern52(variance=var_f, lengthscale=len_f)
lik = newt.likelihoods.Bernoulli(link='logit')
# model = newt.models.GP(kernel=kern, likelihood=lik, X=x, Y=y)
model = newt.models.MarkovGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = newt.models.InfiniteHorizonGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = newt.models.SparseInfiniteHorizonGP(kernel=kern, likelihood=lik, X=x, Y=y, Z=z)

# inf = newt.inference.VariationalInference()
# inf = newt.inference.ExpectationPropagation(power=0.5)
inf = newt.inference.PosteriorLinearisation()
# inf = newt.inference.Laplace()
# inf = newt.inference.LaplaceQuasiNewton(num_data=N, dim=model.func_dim)

trainable_vars = model.vars() + inf.vars()
energy = objax.GradValues(inf.energy, trainable_vars)

lr_adam = 0.1
lr_newton = 1
iters = 20
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
nlpd = model.negative_log_predictive_density(X=x_test, Y=y_test)
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))
print('nlpd: %2.3f' % nlpd)
lb = posterior_mean - 1.96 * posterior_var ** 0.5
ub = posterior_mean + 1.96 * posterior_var ** 0.5
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
