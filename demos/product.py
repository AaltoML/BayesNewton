import bayesnewton
import objax
import numpy as np
import matplotlib.pyplot as plt
import time

N = 1000
XAll = np.linspace(0., 200., num=N)

subband_lengthscales = np.array([500.])
subband_frequencies = np.array([0.2])
modulator_lengthscales = np.array([3.])
modulator_variances = np.array([2.])
subband_kernel = bayesnewton.kernels.SubbandMatern32
modulator_kernel = bayesnewton.kernels.Matern52
kern = bayesnewton.kernels.SpectroTemporal(
    subband_lengthscales=subband_lengthscales,
    subband_frequencies=subband_frequencies,
    modulator_lengthscales=modulator_lengthscales,
    modulator_variances=modulator_variances,
    subband_kernel=subband_kernel,
    modulator_kernel=modulator_kernel
)

lik_var = 0.1
lik = bayesnewton.likelihoods.AudioAmplitudeDemodulation(
    num_components=1,
    variance=lik_var,
    # fix_variance=True
)

dummy_model = bayesnewton.basemodels.MarkovGaussianProcess(kernel=kern, likelihood=lik, X=XAll, Y=np.zeros_like(XAll))
f_samp = np.squeeze(dummy_model.prior_sample(seed=99))
f0true = f_samp[:, 0]
f1true = f_samp[:, 1]
FAll = f0true * lik.link_fn(f1true)
np.random.seed(99)
YAll = FAll + np.random.normal(0., lik_var, N)

# plt.plot(X, f_samp[:, 0], 'b-', linewidth=0.5)
# plt.plot(X, lik.link_fn(f_samp[:, 1]), 'r-', linewidth=0.5)
# plt.plot(X, Y, 'k.', markersize=2)
# plt.show()

# test_start, test_end = 250, 400
# X = np.concatenate([XAll[:test_start], XAll[test_end:]], axis=0)
# Y = np.concatenate([YAll[:test_start], YAll[test_end:]], axis=0)

x_plot = np.linspace(np.min(XAll) - 5, np.max(XAll) + 5, 500)

np.random.seed(123)
# 4-fold cross-validation setup
ind_shuffled = np.random.permutation(N)
ind_split = np.stack(np.split(ind_shuffled, 4))  # 4 random batches of data indices
fold = 3

# Get training and test indices
ind_test = ind_split[fold]  # np.sort(ind_shuffled[:N//4])
ind_train = np.concatenate(ind_split[np.arange(4) != fold])
X = XAll[ind_train]  # 75/25 train/test split
XT = XAll[ind_test]
Y = YAll[ind_train]
YT = YAll[ind_test]

# model = bayesnewton.models.MarkovNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model = bayesnewton.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=X, Y=Y, power=0.5)
# model = bayesnewton.models.MarkovPosteriorLinearisation2ndOrderGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model = bayesnewton.models.InfiniteHorizonVariationalGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model = bayesnewton.models.SparseInfiniteHorizonExpectationPropagationGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# --- Gauss-Newton ---
model = bayesnewton.models.MarkovGaussNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model = bayesnewton.models.MarkovVariationalGaussNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model = bayesnewton.models.MarkovPosteriorLinearisation2ndOrderGaussNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# --- quasi-Newton ---
# model = bayesnewton.models.MarkovQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model = bayesnewton.models.MarkovVariationalQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model = bayesnewton.models.MarkovExpectationPropagationQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y, power=0.5)
# model = bayesnewton.models.MarkovPosteriorLinearisation2ndOrderQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# --- Riemannian grads ---
# model = bayesnewton.models.MarkovVariationalRiemannGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model = bayesnewton.models.MarkovExpectationPropagationRiemannGP(kernel=kern, likelihood=lik, X=X, Y=Y, power=0.5)


lr_adam = 0.01
lr_newton = 0.1
damping = 0.5
iters = 300
opt_hypers = objax.optimizer.Adam(model.vars())
energy = objax.GradValues(model.energy, model.vars())


@objax.Function.with_vars(model.vars() + opt_hypers.vars())
def train_op():
    _, diffs = model.inference(lr=lr_newton, damping=damping)  # perform inference and update variational params
    dE, E = energy()  # compute energy and its gradients w.r.t. hypers
    # opt_hypers(lr_adam, dE)
    test_nlpd_ = model.negative_log_predictive_density(X=XT, Y=YT)
    return E, dE, test_nlpd_, diffs


train_op = objax.Jit(train_op)

t0 = time.time()
for i in range(1, iters + 1):
    loss, grads, test_nlpd, (diff1, diff2) = train_op()
    # if i > 1000:
    #     opt_hypers(lr_adam, grads)
    print('iter %2d, energy: %1.4f, nlpd: %1.4f' % (i, loss[0], test_nlpd))
    # print(diff1, diff2)
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

t0 = time.time()
posterior_mean, posterior_var = model.predict(X=x_plot)
posterior_mean_all, _ = model.predict(X=XAll)
nlpd = model.negative_log_predictive_density(X=XT, Y=YT)
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))
print('NLPD: %1.2f' % nlpd)

posterior_mean_subbands = posterior_mean[:, :1]
posterior_mean_modulators = lik.link_fn(posterior_mean[:, 1:])
posterior_mean_sig = posterior_mean_subbands * posterior_mean_modulators
posterior_var_subbands = posterior_var[:, :1]
posterior_var_modulators = lik.link_fn(posterior_var[:, 1:])
lb_subbands = np.squeeze(posterior_mean_subbands - 1.96 * posterior_var_subbands ** 0.5)
ub_subbands = np.squeeze(posterior_mean_subbands + 1.96 * posterior_var_subbands ** 0.5)
lb_modulators = np.squeeze(lik.link_fn(posterior_mean[:, 1:] - 1.96 * posterior_var[:, 1:] ** 0.5))
ub_modulators = np.squeeze(lik.link_fn(posterior_mean[:, 1:] + 1.96 * posterior_var[:, 1:] ** 0.5))
# lb_sig = lb_subbands * lb_modulators
# ub_sig = ub_subbands * ub_modulators

f0predict = posterior_mean_all[:, 0]
f1predict = posterior_mean_all[:, 1]
f0rmse = np.sqrt(np.mean((f0true-f0predict)**2))
f1rmse = np.sqrt(np.mean((f1true-f1predict)**2))
print('subband RMSE: ', f0rmse)
print('modulator RMSE: ', f1rmse)

print('plotting ...')
plt.figure(1, figsize=(12, 8))
plt.clf()
plt.subplot(2, 1, 1)
plt.title('Amplitude Demodulation (Product Model)')
plt.plot(XAll, FAll, 'c--', linewidth=1.)
plt.plot(x_plot, posterior_mean_sig, 'c', linewidth=1.)
# plt.fill_between(x_plot, lb_sig, ub_sig, color='c', alpha=0.05, label='95% confidence')
plt.plot(X, Y, 'k.', markersize=2, label='train')
plt.plot(XT, YT, 'r.', markersize=2, label='test')
plt.xlim(x_plot[0], x_plot[-1])
plt.gca().xaxis.set_ticklabels([])
plt.subplot(2, 1, 2)
plt.plot(XAll, f_samp[:, 0], 'b--', linewidth=0.5)
plt.plot(XAll, lik.link_fn(f_samp[:, 1]), 'r--', linewidth=0.5)
plt.plot(x_plot, posterior_mean_subbands, 'b-', linewidth=0.5)
# plt.fill_between(x_plot, lb_subbands, ub_subbands, color='b', alpha=0.05)
plt.plot(x_plot, posterior_mean_modulators, 'r-', linewidth=0.5)
# plt.fill_between(x_plot, lb_modulators, ub_modulators, color='r', alpha=0.05)
plt.xlim(x_plot[0], x_plot[-1])
plt.legend()
plt.xlabel('time')
plt.show()
