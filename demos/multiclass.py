import bayesnewton
import objax
import numpy as np
import matplotlib.pyplot as plt
import time

N = 250
XAll = np.linspace(-10., 10., num=N)

num_classes = 3
basekernel = bayesnewton.kernels.Matern52
kern = bayesnewton.kernels.Independent(
    kernels=[basekernel() for i in range(num_classes)]
)

lik = bayesnewton.likelihoods.Softmax(num_classes)

dummy_model = bayesnewton.basemodels.MarkovGaussianProcess(kernel=kern, likelihood=lik, X=XAll, Y=np.zeros_like(XAll))
f_samp = np.squeeze(dummy_model.prior_sample(seed=12345))

# Hard max observation
Y_max = np.argmax(f_samp, 1).reshape(-1,).astype(int)

# One-hot encoding
Y_hot = np.zeros((N, num_classes), dtype=bool)
Y_hot[np.arange(N), Y_max] = 1
order = np.argsort(XAll.reshape(-1,))

colors = ['r', 'b', 'g', 'y']
# plt.figure(1)
# for c in range(num_classes):
#     plt.plot(XAll[order], f_samp[order, c], ".", color=colors[c], label=str(c))
#     plt.plot(XAll[order], Y_hot[order, c], "-", color=colors[c])
# # plt.plot(XAll, f_samp)
# plt.show()

x_plot = np.linspace(np.min(XAll) - 2, np.max(XAll) + 2, 500)

np.random.seed(123)
# 10-fold cross-validation setup
ind_shuffled = np.random.permutation(N)
ind_split = np.stack(np.split(ind_shuffled, 10))  # 10 random batches of data indices
fold = 7

# Get training and test indices
ind_test = ind_split[fold]  # np.sort(ind_shuffled[:N//10])
ind_train = np.concatenate(ind_split[np.arange(10) != fold])
X = XAll[ind_train]  # 90/10 train/test split
XT = XAll[ind_test]
Y = Y_max[ind_train]
YT = Y_max[ind_test]

# model = bayesnewton.models.MarkovNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model = bayesnewton.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=X, Y=Y, power=0.5)
# model = bayesnewton.models.MarkovPosteriorLinearisation2ndOrderGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model = bayesnewton.models.InfiniteHorizonVariationalGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model = bayesnewton.models.SparseInfiniteHorizonExpectationPropagationGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# --- Gauss-Newton ---
# model = bayesnewton.models.MarkovGaussNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
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
iters = 500
opt_hypers = objax.optimizer.Adam(model.vars())
energy = objax.GradValues(model.energy, model.vars())

damping = 0.5


@objax.Function.with_vars(model.vars() + opt_hypers.vars())
def train_op():
    model.inference(lr=lr_newton, damping=damping, ensure_psd=False)  # perform inference and update variational params
    dE, E = energy()  # compute energy and its gradients w.r.t. hypers
    # opt_hypers(lr_adam, dE)
    test_nlpd_ = model.negative_log_predictive_density(X=XT, Y=YT)
    return E, test_nlpd_


train_op = objax.Jit(train_op)

t0 = time.time()
for i in range(1, iters + 1):
    loss, test_nlpd = train_op()
    print('iter %2d, energy: %1.4f, nlpd: %1.4f' % (i, loss[0], test_nlpd))
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

t0 = time.time()
posterior_mean, posterior_var = model.predict(X=x_plot)
nlpd = model.negative_log_predictive_density(X=XT, Y=YT)
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))
print('NLPD: %1.2f' % nlpd)

plt.figure(1)
for c in range(num_classes):
    plt.plot(XAll, f_samp[:, c], ".", color=colors[c], label=str(c))
    plt.plot(XAll, Y_hot[:, c], "-", color=colors[c])
# plt.plot(XAll, f_samp)

plt.figure(2)
for c in range(num_classes):
    plt.plot(x_plot, posterior_mean[:, c], "-", color=colors[c], label=str(c))
plt.show()

# print('plotting ...')
# plt.figure(1, figsize=(12, 8))
# plt.clf()
# plt.subplot(2, 1, 1)
# plt.title('Multi Class Classification')
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
