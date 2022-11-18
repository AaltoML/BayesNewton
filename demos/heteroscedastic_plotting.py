import bayesnewton
import objax
import numpy as np
from jax import vmap
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
import tikzplotlib

maketikz = True

print('loading data ...')
D = np.loadtxt('../data/mcycle.csv', delimiter=',')
X = D[:, 1:2]
Y = D[:, 2:]

# Standardize
X_scaler = StandardScaler().fit(X)
y_scaler = StandardScaler().fit(Y)
Xall = X_scaler.transform(X)
Yall = y_scaler.transform(Y)
x_plot = np.linspace(np.min(Xall)-0.2, np.max(Xall)+0.2, 200)

# Load cross-validation indices
cvind = np.loadtxt('../experiments/motorcycle/cvind.csv').astype(int)

# 10-fold cross-validation setup
nt = np.floor(cvind.shape[0]/10).astype(int)
cvind = np.reshape(cvind[:10*nt], (10, nt))

np.random.seed(123)
fold = 2

# Get training and test indices
test = cvind[fold, :]
train = np.setdiff1d(cvind, test)

# Set training and test data
X = Xall  # [train, :]
Y = Yall  # [train, :]
XT = Xall[test, :]
YT = Yall[test, :]
N = X.shape[0]
M = 20
batch_size = N  # 100
Z = np.linspace(np.min(Xall), np.max(Xall), M)

var_f1 = 1.  # GP variance
len_f1 = 0.5  # GP lengthscale
var_f2 = 1.  # GP variance
len_f2 = 1.  # GP lengthscale

kern1 = bayesnewton.kernels.Matern32(variance=var_f1, lengthscale=len_f1)
kern2 = bayesnewton.kernels.Matern32(variance=var_f2, lengthscale=len_f2)
kern = bayesnewton.kernels.Independent([kern1, kern2])
lik = bayesnewton.likelihoods.HeteroscedasticNoise()

# model = bayesnewton.models.MarkovNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model = bayesnewton.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=X, Y=Y, power=0.5)
# model = bayesnewton.models.MarkovPosteriorLinearisation2ndOrderGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model = bayesnewton.models.InfiniteHorizonVariationalGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model = bayesnewton.models.SparseInfiniteHorizonExpectationPropagationGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# --- Gauss-Newton ---
# model_gn = bayesnewton.models.MarkovGaussNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
model_gn = bayesnewton.models.MarkovVariationalGaussNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model_gn = bayesnewton.models.MarkovPosteriorLinearisation2ndOrderGaussNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# --- quasi-Newton ---
# model_qn = bayesnewton.models.MarkovQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
model_qn = bayesnewton.models.MarkovVariationalQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model_qn = bayesnewton.models.MarkovExpectationPropagationQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y, power=0.5)
# model_qn = bayesnewton.models.MarkovPosteriorLinearisation2ndOrderQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# --- Riemannian grads ---
# model_psd = bayesnewton.models.MarkovNewtonRiemannGP(kernel=kern, likelihood=lik, X=X, Y=Y)
model_psd = bayesnewton.models.MarkovVariationalRiemannGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model_psd = bayesnewton.models.MarkovExpectationPropagationRiemannGP(kernel=kern, likelihood=lik, X=X, Y=Y, power=0.5)


lr_adam = 0.01
lr_newton = 0.3
iters = 300
opt_hypers = objax.optimizer.Adam(model.vars())
opt_hypers_gn = objax.optimizer.Adam(model_gn.vars())
opt_hypers_qn = objax.optimizer.Adam(model_qn.vars())
opt_hypers_psd = objax.optimizer.Adam(model_psd.vars())
energy = objax.GradValues(model.energy, model.vars())
energy_gn = objax.GradValues(model_gn.energy, model_gn.vars())
energy_qn = objax.GradValues(model_qn.energy, model_qn.vars())
energy_psd = objax.GradValues(model_psd.energy, model_psd.vars())


@objax.Function.with_vars(model.vars() + opt_hypers.vars())
def train_op():
    model.inference(lr=lr_newton)  # perform inference and update variational params
    dE, E = energy()  # compute energy and its gradients w.r.t. hypers
    # opt_hypers(lr_adam, dE)
    return E


@objax.Function.with_vars(model_gn.vars() + opt_hypers_gn.vars())
def train_op_gn():
    model_gn.inference(lr=lr_newton)  # perform inference and update variational params
    dE, E = energy_gn()  # compute energy and its gradients w.r.t. hypers
    # opt_hypers_gn(lr_adam, dE)
    return E


@objax.Function.with_vars(model_qn.vars() + opt_hypers_qn.vars())
def train_op_qn():
    model_qn.inference(lr=lr_newton, damping=0.5)  # perform inference and update variational params
    dE, E = energy_qn()  # compute energy and its gradients w.r.t. hypers
    # opt_hypers_qn(lr_adam, dE)
    return E


@objax.Function.with_vars(model_psd.vars() + opt_hypers_psd.vars())
def train_op_psd():
    model_psd.inference(lr=lr_newton)  # perform inference and update variational params
    dE, E = energy_psd()  # compute energy and its gradients w.r.t. hypers
    # opt_hypers_psd(lr_adam, dE)
    return E


train_op = objax.Jit(train_op)
train_op_gn = objax.Jit(train_op_gn)
train_op_qn = objax.Jit(train_op_qn)
train_op_psd = objax.Jit(train_op_psd)

t0 = time.time()
for i in range(1, iters + 1):
    loss = train_op()
    loss_gn = train_op_gn()
    loss_qn = train_op_qn()
    # loss_psd = train_op_psd()
    print('iter %2d, energy: %1.4f' % (i, loss[0]))
    print('iter %2d, gn energy: %1.4f' % (i, loss_gn[0]))
    print('iter %2d, qn energy: %1.4f' % (i, loss_qn[0]))
    # print('iter %2d, psd energy: %1.4f' % (i, loss_psd[0]))
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

t0 = time.time()
posterior_mean, posterior_var = model.predict(X=x_plot)
posterior_mean_gn, posterior_var_gn = model_gn.predict(X=x_plot)
posterior_mean_qn, posterior_var_qn = model_qn.predict(X=x_plot)
# posterior_mean_psd, posterior_var_psd = model_psd.predict(X=x_plot)
nlpd = model.negative_log_predictive_density(X=XT, Y=YT)
nlpd_gn = model_gn.negative_log_predictive_density(X=XT, Y=YT)
nlpd_qn = model_qn.negative_log_predictive_density(X=XT, Y=YT)
# nlpd_psd = model_psd.negative_log_predictive_density(X=XT, Y=YT)
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))
print('NLPD: %1.2f' % nlpd)
print('NLPD G-N: %1.2f' % nlpd_gn)
print('NLPD Q-N: %1.2f' % nlpd_qn)
# print('NLPD PSD: %1.2f' % nlpd_psd)


x_pred = X_scaler.inverse_transform(x_plot)
link = model.likelihood.link_fn
lb = posterior_mean[:, 0] - np.sqrt(posterior_var[:, 0, 0] + link(posterior_mean[:, 1]) ** 2) * 1.96
lb_gn = posterior_mean_gn[:, 0] - np.sqrt(posterior_var_gn[:, 0, 0] + link(posterior_mean_gn[:, 1]) ** 2) * 1.96
lb_qn = posterior_mean_qn[:, 0] - np.sqrt(posterior_var_qn[:, 0, 0] + link(posterior_mean_qn[:, 1]) ** 2) * 1.96
# lb_psd = posterior_mean_psd[:, 0] - np.sqrt(posterior_var_psd[:, 0, 0] + link(posterior_mean_psd[:, 1]) ** 2) * 1.96
ub = posterior_mean[:, 0] + np.sqrt(posterior_var[:, 0, 0] + link(posterior_mean[:, 1]) ** 2) * 1.96
ub_gn = posterior_mean_gn[:, 0] + np.sqrt(posterior_var_gn[:, 0, 0] + link(posterior_mean_gn[:, 1]) ** 2) * 1.96
ub_qn = posterior_mean_qn[:, 0] + np.sqrt(posterior_var_qn[:, 0, 0] + link(posterior_mean_qn[:, 1]) ** 2) * 1.96
# ub_psd = posterior_mean_psd[:, 0] + np.sqrt(posterior_var_psd[:, 0, 0] + link(posterior_mean_psd[:, 1]) ** 2) * 1.96
post_mean = y_scaler.inverse_transform(posterior_mean[:, 0])
post_mean_gn = y_scaler.inverse_transform(posterior_mean_gn[:, 0])
post_mean_qn = y_scaler.inverse_transform(posterior_mean_qn[:, 0])
# post_mean_psd = y_scaler.inverse_transform(posterior_mean_psd[:, 0])
lb = y_scaler.inverse_transform(lb)
lb_gn = y_scaler.inverse_transform(lb_gn)
lb_qn = y_scaler.inverse_transform(lb_qn)
# lb_psd = y_scaler.inverse_transform(lb_psd)
ub = y_scaler.inverse_transform(ub)
ub_gn = y_scaler.inverse_transform(ub_gn)
ub_qn = y_scaler.inverse_transform(ub_qn)
# ub_psd = y_scaler.inverse_transform(ub_psd)

# gauss_plot_indices = np.array([5, 35, 65, 95, 125])
# gauss_plot_indices = np.array([10, 50, 90, 140, 180])
# gauss_plot_indices = np.array([20, 70, 120, 170])
gauss_plot_indices = np.array([8, 50, 85, 112])

print('plotting ...')
plt.figure(1, figsize=(12, 4.2))
plt.clf()
plt.plot(X_scaler.inverse_transform(X), y_scaler.inverse_transform(Y), 'k.', label='training data', markersize=2)
# plt.plot(X_scaler.inverse_transform(XT), y_scaler.inverse_transform(YT), 'r.', label='test')
plt.plot(x_pred, post_mean, 'k', label='posterior mean (heuristic VI)')
plt.fill_between(x_pred, lb, ub, color='k', alpha=0.05, label='95\% confidence')
plt.plot(x_pred, post_mean_gn, 'c-.', label='variational Gauss-Newton')
plt.plot(x_pred, lb_gn, 'c-.', alpha=0.5)
plt.plot(x_pred, ub_gn, 'c-.', alpha=0.5)
plt.plot(x_pred, post_mean_qn, 'r-.', label='variational quasi-Newton')
plt.plot(x_pred, lb_qn, 'r-.', alpha=0.5)
plt.plot(x_pred, ub_qn, 'r-.', alpha=0.5)
# plt.plot(x_pred, post_mean_psd, 'b--', label='posterior mean (Riemann)')
# plt.plot(x_pred, lb_psd, 'b--', alpha=0.5)
# plt.plot(x_pred, ub_psd, 'b--,', alpha=0.5)
plt.vlines((X_scaler.inverse_transform(X))[gauss_plot_indices], -180, 110)
# plt.vlines(x_pred[gauss_plot_indices], -180, 110, linewidths=1)
plt.xlim(x_pred[0], x_pred[-1])
plt.ylim(-180, 110)
if hasattr(model, 'Z'):
    plt.plot(X_scaler.inverse_transform(model.Z.value[:, 0]),
             (np.min(lb)-5)*np.ones_like(model.Z.value[:, 0]),
             'c^',
             markersize=4)
plt.legend()
plt.title('Heteroscedastic Noise Model (Motorcycle Crash Data)')
plt.xlabel('time (milliseconds)')
plt.ylabel('accelerometer reading')
plt.gca().tick_params(axis='both', direction='in')
if maketikz:
    tikzplotlib.save('/Users/wilkinw1/postdoc/inprogress/newton-smoothers/paper/fig/hsced-demo.tex',
                     axis_width='\\figurewidth',
                     axis_height='\\figureheight',
                     tex_relative_path_to_data='./fig/')

# _, jacobian, hessian_true = vmap(model.likelihood.variational_expectation, (0, 0, 0, None))(
#     model.Y,
#     model.posterior_mean.value,
#     model.posterior_variance.value,
#     None
# )
# hessian_true = -bayesnewton.utils.ensure_diagonal_positive_precision(-hessian_true)
# pseudo_likelihood_nat1 = (
#     jacobian - hessian_true @ model.posterior_mean.value
# )
# pseudo_likelihood_nat2 = (
#     -hessian_true
# )
# pseudo_likelihood_cov = bayesnewton.utils.inv_vmap(pseudo_likelihood_nat2)
# pseudo_likelihood_mean = pseudo_likelihood_cov @ pseudo_likelihood_nat1


def plot_gauss_contour(index, plot_num):
    print(index)
    plt.figure(plot_num, figsize=(3.6, 3.6))
    plt.clf()
    gauss_plot_ind = np.array([index])
    gauss_plot_posterior_mean = np.squeeze(model.posterior_mean.value[gauss_plot_ind])
    # gauss_plot_posterior_mean = np.squeeze(model.pseudo_likelihood.mean[gauss_plot_ind])
    gauss_plot_posterior_cov = np.squeeze(model.posterior_variance.value[gauss_plot_ind])
    # gauss_plot_posterior_cov = np.squeeze(model.pseudo_likelihood.covariance[gauss_plot_ind])
    gauss_plot_posterior_std = np.max(np.diag(gauss_plot_posterior_cov) ** 0.5)
    num_points = 200
    X_ = np.linspace(gauss_plot_posterior_mean[0] - 2.2 * gauss_plot_posterior_std,
                     gauss_plot_posterior_mean[0] + 2.2 * gauss_plot_posterior_std, num_points)
    Y_ = np.linspace(gauss_plot_posterior_mean[1] - 2.2 * gauss_plot_posterior_std,
                     gauss_plot_posterior_mean[1] + 2.2 * gauss_plot_posterior_std, num_points)
    if plot_num == 2:
        X_ = np.linspace(gauss_plot_posterior_mean[0] - 1.25 * gauss_plot_posterior_std,  # squeeze manually for visualisation purposes
                         gauss_plot_posterior_mean[0] + 1.25 * gauss_plot_posterior_std, num_points)
        Y_ = np.linspace(gauss_plot_posterior_mean[1] - 2.5 * gauss_plot_posterior_std,
                         gauss_plot_posterior_mean[1] + 2.2 * gauss_plot_posterior_std, num_points)
    X_, Y_ = np.meshgrid(X_, Y_)
    pos = np.dstack((X_, Y_))
    rv = multivariate_normal(gauss_plot_posterior_mean, gauss_plot_posterior_cov)
    Z_ = rv.pdf(pos)
    plt.contour(X_, Y_, Z_, linestyles='solid', linewidths=1, alpha=0.5, levels=4, colors='k')

    rv = multivariate_normal(np.squeeze(model_gn.posterior_mean.value[gauss_plot_ind]),
                             np.squeeze(model_gn.posterior_variance.value[gauss_plot_ind]))
    Z_ = rv.pdf(pos)
    plt.contour(X_, Y_, Z_, linestyles='dashdot', linewidths=1, levels=4, colors='c')

    rv = multivariate_normal(np.squeeze(model_qn.posterior_mean.value[gauss_plot_ind]),
                             np.squeeze(model_qn.posterior_variance.value[gauss_plot_ind]))
    Z_ = rv.pdf(pos)
    plt.contour(X_, Y_, Z_, linestyles='dashdot', linewidths=1, levels=4, colors='r')

    # rv = multivariate_normal(np.squeeze(model_psd.posterior_mean.value[gauss_plot_ind]),
    #                          np.squeeze(model_psd.posterior_variance.value[gauss_plot_ind]))
    # Z_ = rv.pdf(pos)
    # plt.contour(X_, Y_, Z_, linestyles='dashed', linewidths=1, levels=4, colors='b')
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])
    if maketikz:
        tikzplotlib.save("/Users/wilkinw1/postdoc/inprogress/newton-smoothers/paper/fig/hsced-contour"+str(plot_num-2)+".tex",
                         axis_width='\\figurewidth',
                         axis_height='\\figureheight',
                         tex_relative_path_to_data='./fig/')


for i in range(len(gauss_plot_indices)):
    plot_gauss_contour(gauss_plot_indices[i], i+2)


plt.show()
