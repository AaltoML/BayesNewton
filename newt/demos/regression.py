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


np.random.seed(12345)
N = 100
x = np.linspace(-17, 147, num=N)
y = wiggly_time_series(x)
x_test = np.linspace(np.min(x)-15.0, np.max(x)+15.0, num=500)
# x_test = np.linspace(-32.5, 157.5, num=250)
y_test = wiggly_time_series(x_test)
x_plot = np.linspace(np.min(x)-20.0, np.max(x)+20.0, 200)
M = 20
batch_size = N
z = np.linspace(-30, 155, num=M)
# z = x

var_f = 1.0  # GP variance
len_f = 5.0  # GP lengthscale
var_y = 0.5  # observation noise

kern = newt.kernels.Matern52(variance=var_f, lengthscale=len_f)
lik = newt.likelihoods.Gaussian(variance=var_y)
model = newt.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = newt.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = newt.models.MarkovLaplaceGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = newt.models.SparseVariationalGP(kernel=kern, likelihood=lik, X=x, Y=y, Z=z)
# model = newt.models.SparseMarkovVariationalGP(kernel=kern, likelihood=lik, X=x, Y=y, Z=z)

lr_adam = 0.1
lr_newton = 1
iters = 20
opt_hypers = objax.optimizer.Adam(model.vars())
energy = objax.GradValues(model.energy, model.vars())
inf_args = {
    "power": 0.5,  # the EP power
}


@objax.Function.with_vars(model.vars() + opt_hypers.vars())
def train_op():
    # batch = np.random.permutation(N)[:batch_size]
    # model.inference(lr=lr_newton, batch_ind=batch, **inf_args)  # perform inference and update variational params
    # dE, E = energy(batch_ind=batch, **inf_args)  # compute energy and its gradients w.r.t. hypers
    model.inference(lr=lr_newton, **inf_args)  # perform inference and update variational params
    dE, E = energy(**inf_args)  # compute energy and its gradients w.r.t. hypers
    opt_hypers(lr_adam, dE)
    return E


train_op = objax.Jit(train_op)

t0 = time.time()
for i in range(1, iters + 1):
    loss = train_op()
    print('iter %2d, energy: %1.4f' % (i, loss[0]))
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

# posterior_samples = model.posterior_sample(X=x_plot, num_samps=20)  # only implemented for Markov GPs

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
# plt.plot(x_plot, posterior_samples.T, 'b', alpha=0.2)
plt.fill_between(x_plot, lb, ub, color='b', alpha=0.05, label='95% confidence')
plt.xlim([x_plot[0], x_plot[-1]])
if hasattr(model, 'Z'):
    plt.plot(model.Z.value[:, 0], -2 * np.ones_like(model.Z.value[:, 0]), 'b^', markersize=5)
# plt.xlim([x_test[0], x_test[-1]])
# plt.ylim([-2, 5])
plt.legend()
plt.title('GP regression')
plt.xlabel('$X$')
plt.show()
