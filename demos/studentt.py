import bayesnewton
import objax
from jax import vmap
import numpy as np
import matplotlib.pyplot as plt
import time


def wiggly_time_series(x_):
    noise_var = 0.2  # true observation noise scale
    return (np.cos(0.04*x_+0.33*np.pi) * np.sin(0.2*x_) +
            np.sqrt(noise_var) * np.random.standard_t(3., x_.shape) +
            0.0 * x_)


np.random.seed(123)
N = 100
x = np.linspace(-17, 147, num=N)
y = wiggly_time_series(x)
x_test = np.linspace(np.min(x)-10.0, np.max(x)+10.0, num=500)
y_test = wiggly_time_series(x_test)
x_plot = np.linspace(np.min(x)-20.0, np.max(x)+20.0, 200)

var_f = 1.0  # GP variance
len_f = 5.0  # GP lengthscale
var_y = 0.2  # observation noise

kern = bayesnewton.kernels.Matern52(variance=var_f, lengthscale=len_f)
lik = bayesnewton.likelihoods.StudentsT(scale=0.5, df=3.)
# lik = bayesnewton.likelihoods.Gaussian(variance=0.5)

# model = bayesnewton.models.MarkovNewtonGP(kernel=kern, likelihood=lik, X=x, Y=y)
model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=x, Y=y, power=0.5)
# -- Gauss-Newton ---
# model = bayesnewton.models.MarkovGaussNewtonGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.MarkovVariationalGaussNewtonGP(kernel=kern, likelihood=lik, X=x, Y=y)
# -- quasi-Newton ---
# model = bayesnewton.models.MarkovQuasiNewtonGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.MarkovVariationalQuasiNewtonGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.MarkovExpectationPropagationQuasiNewtonGP(kernel=kern, likelihood=lik, X=x, Y=y, power=0.5)
# --- Riemannian grads ---
# model = bayesnewton.models.MarkovVariationalRiemannGP(kernel=kern, likelihood=lik, X=x, Y=y)

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
nlpd = model.negative_log_predictive_density(X=x_test, Y=y_test)
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))
print('nlpd: %2.3f' % nlpd)
lb = posterior_mean - 1.96 * posterior_var ** 0.5
ub = posterior_mean + 1.96 * posterior_var ** 0.5

_, _, hessian = vmap(model.likelihood.log_likelihood_gradients)(  # parallel
    model.Y,
    model.posterior_mean.value
)

outliers = np.argwhere(np.squeeze(hessian > 0))

print('plotting ...')
plt.figure(1, figsize=(12, 5))
plt.clf()
plt.plot(x, y, 'k.', label='training observations')
plt.plot(x_test, y_test, 'r.', alpha=0.4, label='test observations')
plt.plot(x_plot, posterior_mean, 'b', label='posterior mean')
plt.plot(x[outliers], y[outliers], 'g*', label='outliers')
plt.fill_between(x_plot, lb, ub, color='b', alpha=0.05, label='95% confidence')
plt.xlim([x_plot[0], x_plot[-1]])
if hasattr(model, 'Z'):
    plt.plot(model.Z.value[:, 0], -2 * np.ones_like(model.Z.value[:, 0]), 'b^', markersize=5)
plt.legend()
plt.title('Robust GP regression (Students\' t likelihood)')
plt.xlabel('$X$')
plt.show()
