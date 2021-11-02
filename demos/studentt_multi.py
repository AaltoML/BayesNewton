import bayesnewton
import objax
from jax import vmap
import numpy as np
import matplotlib.pyplot as plt
import time


def wiggly_time_series_1(x_):
    return np.cos(0.04*x_+0.33*np.pi) * np.sin(0.2*x_)


def wiggly_time_series_2(x_):
    return np.cos(0.05*x_+0.4*np.pi) * np.sin(0.1*x_)
    # return 2 * wiggly_time_series_1(x_)


def wiggly_time_series_3(x_):
    # return np.cos(0.05*x_+0.4*np.pi) * np.sin(0.1*x_)
    return 2 * wiggly_time_series_1(x_)


np.random.seed(123)
N = 250
x = np.linspace(-17, 147, num=N)
# x = np.concatenate([
#     np.linspace(-17, 55, num=N),
#     np.linspace(75, 147, num=N)
#     ], axis=0)
f1 = wiggly_time_series_1(x)[:, None]
f2 = wiggly_time_series_2(x)[:, None]
f3 = wiggly_time_series_3(x)[:, None]
f = np.concatenate([f1, f2, f3], axis=1)[..., None]

# noise_cov = np.array([[0.2, 0.1], [0.1, 0.3]])
# noise_cov = np.array([[0.1, -0.075], [-0.075, 0.2]])
noise_cov = np.array([[0.1, -0.075, -0.025], [-0.075, 0.2, 0.05], [-0.025, 0.05, 0.3]])

noise = np.linalg.cholesky(noise_cov)[None] @ np.random.standard_t(3., f.shape)
y = f + noise

# plt.figure(1)
# plt.plot(x, f1, 'b-')
# plt.plot(x, f2, 'r-')
# plt.plot(x, y[:, 0], 'b.')
# plt.plot(x, y[:, 1], 'r.')
#
# plt.figure(2)
# plt.plot(x, noise[:, 0], 'b')
# plt.plot(x, noise[:, 1], 'r')
# plt.show()

# x_test = np.linspace(np.min(x)-10.0, np.max(x)+10.0, num=500)
# f1_test = wiggly_time_series_1(x_test)[:, None]
# f2_test = wiggly_time_series_2(x_test)[:, None]
# f_test = np.concatenate([f1_test, f2_test], axis=1)[..., None]
# noise_test = np.linalg.cholesky(noise_cov)[None] @ np.random.standard_t(3., f_test.shape)
# y_test = f_test + noise_test
x_plot = np.linspace(np.min(x)-20.0, np.max(x)+20.0, 200)
f1_plot = wiggly_time_series_1(x_plot)
f2_plot = wiggly_time_series_2(x_plot)
f3_plot = wiggly_time_series_3(x_plot)

fold = 1

np.random.seed(123)
# 4-fold cross-validation setup
ind_shuffled = np.random.permutation((y.flatten().shape[0] // 4) * 4)
ind_split = np.stack(np.split(ind_shuffled, 4))  # 4 random batches of data indices

# Get training and test indices
ind_test = ind_split[fold]  # np.sort(ind_shuffled[:N//4])
ind_train = np.concatenate(ind_split[np.arange(4) != fold])
X, XT = x, x
Y, YT = y.flatten(), y.flatten()
Y[ind_test] = np.nan  # 75/25 train/test split
YT[ind_train] = np.nan
Y = Y.reshape(N, -1)
YT = YT.reshape(N, -1)

var_f = 1.0  # GP variance
len_f = 15.0  # GP lengthscale

kern1 = bayesnewton.kernels.Matern52(variance=var_f, lengthscale=len_f)
kern2 = bayesnewton.kernels.Matern52(variance=var_f, lengthscale=len_f)
kern3 = bayesnewton.kernels.Matern52(variance=var_f, lengthscale=len_f)
kern = bayesnewton.kernels.Independent(kernels=[kern1, kern2, kern3])
lik = bayesnewton.likelihoods.StudentsTMultivariate(scale=noise_cov, df=3.)
# lik = bayesnewton.likelihoods.Gaussian(variance=0.5)

# model = bayesnewton.models.MarkovNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model = bayesnewton.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=X, Y=Y, power=0.5)
# -- Gauss-Newton ---
# model = bayesnewton.models.MarkovGaussNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model = bayesnewton.models.MarkovVariationalGaussNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# -- quasi-Newton ---
# model = bayesnewton.models.MarkovQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
model = bayesnewton.models.MarkovVariationalQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model = bayesnewton.models.MarkovExpectationPropagationQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y, power=0.5)
# --- Riemannian grads ---
# model = bayesnewton.models.MarkovVariationalRiemannGP(kernel=kern, likelihood=lik, X=X, Y=Y)

# --- GP ---
# model = bayesnewton.models.VariationalGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model = bayesnewton.models.VariationalQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# --- sparse ---
# model = bayesnewton.models.SparseVariationalGP(kernel=kern, likelihood=lik, X=X, Y=Y, Z=Z)
# --- sparse quasi-Newton ---
# model = bayesnewton.models.SparseQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y, Z=Z)
# model = bayesnewton.models.SparseVariationalQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y, Z=Z)

lr_adam = 0.1
lr_newton = 0.3
iters = 300
opt_hypers = objax.optimizer.Adam(model.vars())
energy = objax.GradValues(model.energy, model.vars())

unscented_transform = bayesnewton.cubature.Unscented(dim=3)  # 5th-order unscented transform

damping = 0.5


# @objax.Function.with_vars(model.vars() + opt_hypers.vars())
# def train_op():
#     model.inference(lr=lr_newton, damping=damping, cubature=unscented_transform)  # perform inference and update variational params
#     dE, E = energy(cubature=unscented_transform)  # compute energy and its gradients w.r.t. hypers
#     opt_hypers(lr_adam, dE)
#     test_nlpd_ = model.negative_log_predictive_density(X=XT, Y=YT)
#     return E, test_nlpd_


@objax.Function.with_vars(model.vars() + opt_hypers.vars())
def train_op_warmup():
    model.inference(lr=lr_newton, damping=damping, cubature=unscented_transform)  # perform inference and update variational params
    dE, E = energy(cubature=unscented_transform)  # compute energy and its gradients w.r.t. hypers
    # opt_hypers(lr_adam, dE)
    test_nlpd_ = model.negative_log_predictive_density(X=XT, Y=YT, cubature=unscented_transform)
    return E, test_nlpd_


# train_op = objax.Jit(train_op)
train_op_warmup = objax.Jit(train_op_warmup)

t0 = time.time()
# for i in range(1, iters // 10 + 1):
for i in range(1, iters + 1):
    loss, test_nlpd = train_op_warmup()
    print('iter %2d, energy: %1.4f, nlpd: %1.4f' % (i, loss[0], test_nlpd))
# for i in range(iters // 10 + 1, iters + 1):
#     loss, test_nlpd = train_op()
#     print('iter %2d, energy: %1.4f, nlpd: %1.4f' % (i, loss[0], test_nlpd))
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

t0 = time.time()
posterior_mean, posterior_cov = model.predict_y(X=x_plot, cubature=unscented_transform)
posterior_mean_f, posterior_cov_f = model.predict(X=x_plot)
nlpd = model.negative_log_predictive_density(X=XT, Y=YT, cubature=unscented_transform)
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))
print('nlpd: %2.3f' % nlpd)
posterior_var = bayesnewton.utils.diag(posterior_cov)
crosscov = posterior_cov_f[:, 0, 1]
print('avg. cross cov', np.mean(np.abs(crosscov)))
lb = posterior_mean - 1.96 * posterior_var ** 0.5
ub = posterior_mean + 1.96 * posterior_var ** 0.5

f0rmse = np.sqrt(np.mean((f1_plot-posterior_mean_f[:, 0])**2))
f1rmse = np.sqrt(np.mean((f2_plot-posterior_mean_f[:, 1])**2))
f2rmse = np.sqrt(np.mean((f3_plot-posterior_mean_f[:, 2])**2))
print('RMSE 1: ', f0rmse)
print('RMSE 2: ', f1rmse)
print('RMSE 3: ', f2rmse)

# _, _, hessian = vmap(model.likelihood.log_likelihood_gradients)(  # parallel
#     model.Y,
#     model.posterior_mean.value
# )
#
# outliers = np.argwhere(np.squeeze(hessian > 0))

print('plotting ...')
plt.figure(1, figsize=(12, 5))
plt.clf()
plt.plot(X, Y[:, 0], 'b.')
plt.plot(X, Y[:, 1], 'r.')
plt.plot(X, Y[:, 2], 'g.')
plt.plot(X, YT[:, 0], 'b*', markersize=4, alpha=0.4)
plt.plot(X, YT[:, 1], 'r*', markersize=4, alpha=0.4)
plt.plot(X, YT[:, 2], 'g*', markersize=4, alpha=0.4)
# plt.plot(x_test, y_test, 'r.', alpha=0.4, label='test observations')
plt.plot(x_plot, posterior_mean[:, 0], 'b')
plt.plot(x_plot, posterior_mean[:, 1], 'r')
plt.plot(x_plot, posterior_mean[:, 2], 'g')
plt.plot(x_plot, f1_plot, 'b--')
plt.plot(x_plot, f2_plot, 'r--')
plt.plot(x_plot, f3_plot, 'g--')
plt.fill_between(x_plot, lb[:, 0], ub[:, 0], color='b', alpha=0.05)
plt.fill_between(x_plot, lb[:, 1], ub[:, 1], color='r', alpha=0.05)
plt.fill_between(x_plot, lb[:, 2], ub[:, 2], color='g', alpha=0.05)
# plt.plot(x[outliers], y[outliers], 'g*', label='outliers')
plt.xlim([x_plot[0], x_plot[-1]])
if hasattr(model, 'Z'):
    plt.plot(model.Z.value[:, 0], -2 * np.ones_like(model.Z.value[:, 0]), 'b^', markersize=5)
plt.legend()
plt.title('Robust GP regression (Student\'s t likelihood)')
plt.xlabel('$X$')
plt.show()
