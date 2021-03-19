import newt
import objax
import numpy as np
import matplotlib.pyplot as plt
import time


def wiggly_time_series(x_):
    noise_var = 0.15  # true observation noise
    return (np.cos(0.04*x_+0.33*np.pi) * np.sin(0.2*x_) +
            np.math.sqrt(noise_var) * np.random.normal(0, 1, x_.shape) +
            0.0 * x_)  # 0.02 * x_)


print('generating some data ...')
np.random.seed(12345)
N = 100
# x0 = np.random.permutation(np.linspace(-25.0, 30.0, num=N//2) + 1*np.random.randn(N//2))  # unevenly spaced
# x1 = np.random.permutation(np.linspace(60.0, 150.0, num=N//2) + 1*np.random.randn(N//2))  # unevenly spaced
# x = np.concatenate([x0, x1], axis=0)
x = np.linspace(-17, 147, num=N)
x = np.sort(x, axis=0)
y = wiggly_time_series(x)
x_test = np.linspace(np.min(x)-15.0, np.max(x)+15.0, num=500)
# x_test = np.linspace(-32.5, 157.5, num=250)
y_test = wiggly_time_series(x_test)
x_plot = np.linspace(np.min(x)-20.0, np.max(x)+20.0, 200)
M = 20
batch_size = N  # TODO: why does using smaller batch_size result in longer compile time?
z = np.linspace(-30, 155, num=M)
# z = x
# z = np.linspace(-10, 140, num=M)


z = z[:, None]
x = x[:, None]
x_plot = x_plot[:, None]

var_f = 1.0  # GP variance
len_f = 5.0  # GP lengthscale
var_y = 0.5  # observation noise

kern = newt.kernels.Matern52(variance=var_f, lengthscale=len_f)
lik = newt.likelihoods.Gaussian(variance=var_y)
# model = newt.models.GP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = newt.models.SparseGP(kernel=kern, likelihood=lik, X=x, Y=y, Z=z, opt_z=True)
model = newt.models.MarkovGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = newt.models.InfiniteHorizonGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = newt.models.SparseMarkovGP(kernel=kern, likelihood=lik, X=x, Y=y, Z=z)
# model = newt.models.SparseInfiniteHorizonGP(kernel=kern, likelihood=lik, X=x, Y=y, Z=z)

inf = newt.inference.VariationalInference()
# inf = newt.inference.Laplace()
# inf = newt.inference.PosteriorLinearisation()
# inf = newt.inference.Taylor()
# inf = newt.inference.ExpectationPropagation(power=0.5)
# inf = newt.inference.LaplaceQuasiNewton(num_data=N, dim=model.func_dim)
# inf = newt.inference.VariationalQuasiNewton(num_data=N, dim=model.func_dim)

trainable_vars = model.vars() + inf.vars()
energy = objax.GradValues(inf.energy, trainable_vars)

lr_adam = 0.1
lr_newton = 1
iters = 20
opt = objax.optimizer.Adam(trainable_vars)


def train_op():
    batch = np.random.permutation(N)[:batch_size]
    inf(model, lr=lr_newton, batch_ind=batch)  # perform inference and update variational params
    dE, E = energy(model, batch_ind=batch)  # compute energy and its gradients w.r.t. hypers
    return dE, E


train_op = objax.Jit(train_op, trainable_vars)

t0 = time.time()
for i in range(1, iters + 1):
    grad, loss = train_op()
    opt(lr_adam, grad)
    print('iter %2d, energy: %1.4f' % (i, loss[0]))
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

t0 = time.time()
posterior_mean, posterior_var = model.predict_y(X=x_plot)
nlpd = model.negative_log_predictive_density(X=x_test, Y=y_test)
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))
print('nlpd: %2.3f' % nlpd)
lb = posterior_mean - 1.96 * posterior_var ** 0.5
ub = posterior_mean + 1.96 * posterior_var ** 0.5

print('plotting ...')
plt.figure(1, figsize=(12, 5))
plt.clf()
plt.plot(x, y, 'k.', label='training observations')
plt.plot(x_test, y_test, 'r.', alpha=0.4, label='test observations')
plt.plot(x_plot, posterior_mean, 'b', label='posterior mean')
plt.fill_between(x_plot[:, 0], lb, ub, color='b', alpha=0.05, label='95% confidence')
# plt.plot(x_plot, posterior_samp, 'b', alpha=0.15)
plt.xlim([x_plot[0], x_plot[-1]])
if hasattr(model, 'Z'):
    plt.plot(model.Z.value[:, 0], -2 * np.ones_like(model.Z.value[:, 0]), 'b^', markersize=5)
# plt.xlim([x_test[0], x_test[-1]])
# plt.ylim([-2, 5])
plt.legend()
plt.title('GP regression')
plt.xlabel('$X$')
plt.show()
