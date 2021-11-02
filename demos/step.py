import bayesnewton
import objax
import numpy as np
import matplotlib.pyplot as plt
import time


def step_function(x_):
    # return np.maximum(np.sign(x_), 0.) - 0.5
    return np.sign(x_)


print('generating some data ...')
np.random.seed(0)
# N = 50
# x = np.linspace(-1, 1, num=N)
# x = np.random.uniform(-1, 1, N)
# y = step_function(x + np.random.randn(*x.shape) * 1e-1)  # + np.random.randn(*x.shape) * 1e-2

N = 40
x = np.linspace(-1, 1, N)[:, None]
# x = np.random.uniform(-1, 1, N)[:, None]
f_step = lambda x_: -1 if x_ < 0 else 1.
y = np.reshape([f_step(x_) for x_ in x], x.shape) + np.random.randn(*x.shape) * 1e-2
x_plot = np.linspace(-2., 2, 300)  # test inputs

# num_low = 25
# num_high = 25
# gap = -.02
# noise = 0.0001
# x = np.vstack((np.linspace(-1, -gap/2.0, num_low)[:, np.newaxis],
#               np.linspace(gap/2.0, 1, num_high)[:, np.newaxis]))
# y = np.vstack((-np.ones((num_low, 1)), np.ones((num_high, 1))))

# x = x[:, None]
x_plot = x_plot[:, None]

var_f = 1.0  # GP variance
len_f = 0.1  # GP lengthscale
var_y = 0.1  # observation noise

kern = bayesnewton.kernels.Matern72(variance=var_f, lengthscale=len_f)
lik = bayesnewton.likelihoods.Gaussian(variance=var_y)
model = bayesnewton.models.VariationalGP(kernel=kern, likelihood=lik, X=x, Y=y)


lr_adam = 0.1
lr_newton = 1.
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

t0 = time.time()
posterior_mean, posterior_var = model.predict_y(X=x_plot)
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))
lb = posterior_mean - 1.96 * posterior_var ** 0.5
ub = posterior_mean + 1.96 * posterior_var ** 0.5

print('plotting ...')
plt.figure(1, figsize=(8, 4))
plt.clf()
plt.plot(x, y, 'k.')  # , label='training observations')
plt.plot(x_plot, posterior_mean, 'b')  # , label='posterior mean')
plt.fill_between(x_plot[:, 0], lb, ub, color='b', alpha=0.05)  # , label='2 std')
plt.xlim([x_plot[0], x_plot[-1]])
plt.ylim([-0.5, 1.5])
# plt.legend(loc=2)
plt.xticks([-2, -1, 0., 1., 2])
plt.yticks([-2, -1., 0., 1., 2])
plt.title('GP regression - Step Function')
plt.xlabel('$X$')
# plt.savefig('/Users/wilkinw1/postdoc/gp_course/lec8_deepgps/step_function.png')
plt.show()
