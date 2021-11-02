import bayesnewton
import objax
import numpy as np
import matplotlib.pyplot as plt
import time
import tikzplotlib
import sys

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

# x_plot = np.linspace(np.min(XAll) - 5, np.max(XAll) + 5, 1000)
x_plot = np.linspace(60, 150, 400)

if len(sys.argv) > 1:
    inf_method = int(sys.argv[1])
    approx_method = int(sys.argv[2])
    fold = int(sys.argv[3])
    plot_final = False
    make_tikz = False
    save_result = True
else:
    inf_method = 1
    approx_method = 0
    fold = 0
    plot_final = True
    make_tikz = True
    save_result = False

np.random.seed(123)
# 4-fold cross-validation setup
ind_shuffled = np.random.permutation(N)
ind_split = np.stack(np.split(ind_shuffled, 4))  # 4 random batches of data indices

# Get training and test indices
ind_test = ind_split[fold]  # np.sort(ind_shuffled[:N//4])
ind_train = np.concatenate(ind_split[np.arange(4) != fold])
X = XAll[ind_train]  # 75/25 train/test split
XT = XAll[ind_test]
Y = YAll[ind_train]
YT = YAll[ind_test]


model0 = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=X, Y=Y)
model1 = bayesnewton.models.MarkovVariationalGaussNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)

lr_adam = 0.01
lr_newton = 0.1
iters = 300
opt_hypers0 = objax.optimizer.Adam(model0.vars())
opt_hypers1 = objax.optimizer.Adam(model1.vars())
energy0 = objax.GradValues(model0.energy, model0.vars())
energy1 = objax.GradValues(model1.energy, model1.vars())

if inf_method == 0:  # Newton
    damping = 0.4
else:
    damping = 0.5


@objax.Function.with_vars(model0.vars() + opt_hypers0.vars())
def train_op0():
    _, diffs = model0.inference(lr=lr_newton, damping=damping)  # perform inference and update variational params
    dE, E = energy0()  # compute energy and its gradients w.r.t. hypers
    # opt_hypers0(lr_adam, dE)
    test_nlpd_ = model0.negative_log_predictive_density(X=XT, Y=YT)
    posterior_mean_all_, _ = model0.predict(X=XAll)
    return E, dE, test_nlpd_, diffs, posterior_mean_all_


@objax.Function.with_vars(model1.vars() + opt_hypers1.vars())
def train_op1():
    _, diffs = model1.inference(lr=lr_newton, damping=damping)  # perform inference and update variational params
    dE, E = energy1()  # compute energy and its gradients w.r.t. hypers
    # opt_hypers1(lr_adam, dE)
    test_nlpd_ = model1.negative_log_predictive_density(X=XT, Y=YT)
    posterior_mean_all_, _ = model1.predict(X=XAll)
    return E, dE, test_nlpd_, diffs, posterior_mean_all_


train_op0 = objax.Jit(train_op0)
train_op1 = objax.Jit(train_op1)

t0 = time.time()
for i in range(1, iters + 1):
    loss, grads, test_nlpd, (diff1, diff2), posterior_mean_all = train_op0()
    loss, grads, test_nlpd, (diff1, diff2), posterior_mean_all = train_op1()
    # if i > 1000:
    #     opt_hypers(lr_adam, grads)
    f0predict = posterior_mean_all[:, 0]
    f1predict = posterior_mean_all[:, 1]
    rmse = np.sqrt((np.mean((f0true-f0predict)**2) + np.mean((f1true-f1predict)**2)) / 2.)
    print('iter %2d, energy: %1.4f, nlpd: %1.4f, rmse: %1.4f' % (i, loss[0], test_nlpd, rmse))
    print(diff1, diff2)
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

if plot_final:
    t0 = time.time()
    posterior_mean0, posterior_var0 = model0.predict(X=x_plot)
    posterior_mean1, posterior_var1 = model1.predict(X=x_plot)
    posterior_mean_all0, _ = model0.predict(X=XAll)
    posterior_mean_all1, _ = model1.predict(X=XAll)
    t1 = time.time()
    print('prediction time: %2.2f secs' % (t1-t0))

    posterior_mean_subbands0 = posterior_mean0[:, :1]
    posterior_mean_subbands1 = posterior_mean1[:, :1]
    posterior_mean_modulators0 = lik.link_fn(posterior_mean0[:, 1:])
    posterior_mean_modulators1 = lik.link_fn(posterior_mean1[:, 1:])
    posterior_mean_sig0 = posterior_mean_subbands0 * posterior_mean_modulators0
    posterior_mean_sig1 = posterior_mean_subbands1 * posterior_mean_modulators1
    posterior_var_subbands0 = posterior_var0[:, :1, :1]
    posterior_var_subbands1 = posterior_var1[:, :1, :1]
    posterior_var_modulators0 = lik.link_fn(posterior_var0[:, 1:, 1:])
    posterior_var_modulators1 = lik.link_fn(posterior_var1[:, 1:, 1:])
    lb_subbands0 = np.squeeze(posterior_mean_subbands0) - 1.96 * np.squeeze(posterior_var_subbands0 ** 0.5)
    lb_subbands1 = np.squeeze(posterior_mean_subbands1) - 1.96 * np.squeeze(posterior_var_subbands1 ** 0.5)
    ub_subbands0 = np.squeeze(posterior_mean_subbands0) + 1.96 * np.squeeze(posterior_var_subbands0 ** 0.5)
    ub_subbands1 = np.squeeze(posterior_mean_subbands1) + 1.96 * np.squeeze(posterior_var_subbands1 ** 0.5)
    lb_modulators0 = lik.link_fn(np.squeeze(posterior_mean0[:, 1:]) - 1.96 * np.squeeze(posterior_var0[:, 1:, 1:] ** 0.5))
    lb_modulators1 = lik.link_fn(np.squeeze(posterior_mean1[:, 1:]) - 1.96 * np.squeeze(posterior_var1[:, 1:, 1:] ** 0.5))
    ub_modulators0 = lik.link_fn(np.squeeze(posterior_mean0[:, 1:]) + 1.96 * np.squeeze(posterior_var0[:, 1:, 1:] ** 0.5))
    ub_modulators1 = lik.link_fn(np.squeeze(posterior_mean1[:, 1:]) + 1.96 * np.squeeze(posterior_var1[:, 1:, 1:] ** 0.5))

    # print('plotting ...')
    # plt.figure(1, figsize=(12, 8))
    # plt.clf()
    # plt.subplot(2, 1, 1)
    # plt.title('Amplitude Demodulation (Product Model)')
    # plt.plot(XAll, FAll, 'c--', linewidth=1.)
    # plt.plot(x_plot, posterior_mean_sig, 'c', linewidth=1.)
    # # plt.fill_between(x_plot, lb_sig, ub_sig, color='c', alpha=0.05, label='95% confidence')
    # plt.plot(X, Y, 'k.', markersize=2, label='train')
    # plt.plot(XT, YT, 'r.', markersize=2, label='test')
    # plt.xlim(x_plot[0], x_plot[-1])
    # plt.gca().xaxis.set_ticklabels([])
    # plt.subplot(2, 1, 2)
    # plt.plot(XAll, f_samp[:, 0], 'b--', linewidth=0.5)
    # plt.plot(XAll, lik.link_fn(f_samp[:, 1]), 'r--', linewidth=0.5)
    # plt.plot(x_plot, posterior_mean_subbands, 'b-', linewidth=0.5)
    # # plt.fill_between(x_plot, lb_subbands, ub_subbands, color='b', alpha=0.05)
    # plt.plot(x_plot, posterior_mean_modulators, 'r-', linewidth=0.5)
    # # plt.fill_between(x_plot, lb_modulators, ub_modulators, color='r', alpha=0.05)
    # plt.xlim(x_plot[0], x_plot[-1])
    # plt.legend()
    # plt.xlabel('time')
    # plt.show()

    plt.figure(1)
    # plt.plot(XAll, FAll, 'k--', linewidth=1., alpha=0.3, label='signal')
    plt.plot(X, Y, 'k.', markersize=1, label='training data')
    # plt.xlim(x_plot[0], x_plot[-1])
    plt.xlim(60, 150)
    plt.ylim(-4.5, 5.)
    plt.legend()
    plt.title('Observed Signal')
    plt.gca().tick_params(axis='both', direction='in')
    tikzplotlib.save(
        "/Users/wilkinw1/postdoc/inprogress/newton-smoothers/paper/fig/product-sig.tex",
        axis_width='\\figurewidth',
        axis_height='\\figureheight',
        tex_relative_path_to_data='./fig/')
    plt.figure(2)
    plt.plot(XAll, f_samp[:, 0], 'k--', linewidth=1, label='ground truth')
    plt.plot(x_plot, posterior_mean_subbands0, 'b-', linewidth=0.5, label='heuristic VI')
    plt.plot(x_plot, posterior_mean_subbands1, 'r-', linewidth=0.5, label='variational Gauss-Newton')
    plt.fill_between(x_plot, lb_subbands0, ub_subbands0, color='b', alpha=0.05)
    plt.fill_between(x_plot, lb_subbands1, ub_subbands1, color='r', alpha=0.05)
    # plt.xlim(x_plot[0], x_plot[-1])
    plt.xlim(60, 150)
    plt.ylim(-2, 2)
    # plt.legend()
    plt.title('Periodic Component')
    plt.gca().tick_params(axis='both', direction='in')
    tikzplotlib.save(
        "/Users/wilkinw1/postdoc/inprogress/newton-smoothers/paper/fig/product-subband.tex",
        axis_width='\\figurewidth',
        axis_height='\\figureheight',
        tex_relative_path_to_data='./fig/')
    plt.figure(3)
    plt.plot(XAll, lik.link_fn(f_samp[:, 1]), 'k--', linewidth=1, label='ground truth')
    plt.plot(x_plot, posterior_mean_modulators0, 'b-', linewidth=0.5, label='heuristic VI')
    plt.plot(x_plot, posterior_mean_modulators1, 'r-', linewidth=0.5, label='variational Gauss-Newton')
    plt.fill_between(x_plot, lb_modulators0, ub_modulators0, color='b', alpha=0.05)
    plt.fill_between(x_plot, lb_modulators1, ub_modulators1, color='r', alpha=0.05)
    # plt.xlim(x_plot[0], x_plot[-1])
    plt.xlim(60, 150)
    plt.ylim(-0.2, 4.)
    plt.xlabel('time')
    plt.title('Amplitude Component')
    plt.legend()
    plt.gca().tick_params(axis='both', direction='in')
    tikzplotlib.save(
        "/Users/wilkinw1/postdoc/inprogress/newton-smoothers/paper/fig/product-modulator.tex",
        axis_width='\\figurewidth',
        axis_height='\\figureheight',
        tex_relative_path_to_data='./fig/')
    plt.show()
