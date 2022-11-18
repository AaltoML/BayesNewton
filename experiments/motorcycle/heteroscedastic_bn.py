import bayesnewton
import objax
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
import pickle
import sys

print('loading data ...')
D = np.loadtxt('../../data/mcycle.csv', delimiter=',')
X = D[:, 1:2]
Y = D[:, 2:]

# Standardize
X_scaler = StandardScaler().fit(X)
y_scaler = StandardScaler().fit(Y)
Xall = X_scaler.transform(X)
Yall = y_scaler.transform(Y)
x_plot = np.linspace(np.min(Xall)-0.2, np.max(Xall)+0.2, 200)

# Load cross-validation indices
# cvind = np.loadtxt('cvind.csv').astype(int)

# 10-fold cross-validation setup
# nt = np.floor(cvind.shape[0]/10).astype(int)
# cvind = np.reshape(cvind[:10*nt], (10, nt))

if len(sys.argv) > 1:
    inf_method = int(sys.argv[1])
    approx_method = int(sys.argv[2])
    fold = int(sys.argv[3])
    plot_final = False
else:
    inf_method = 0
    approx_method = 0
    fold = 0
    plot_final = True

np.random.seed(123)
# 4-fold cross-validation setup
ind_shuffled = np.random.permutation((Y.shape[0] // 4) * 4)
ind_split = np.stack(np.split(ind_shuffled, 4))  # 4 random batches of data indices

# Get training and test indices
ind_test = ind_split[fold]  # np.sort(ind_shuffled[:N//4])
ind_train = np.concatenate(ind_split[np.arange(4) != fold])
X = Xall[ind_train]  # 75/25 train/test split
XT = Xall[ind_test]
Y = Yall[ind_train]
YT = Yall[ind_test]

N = X.shape[0]
M = 20
batch_size = N  # 100
Z = np.linspace(np.min(Xall), np.max(Xall), M)

var_f1 = 1.  # GP variance
len_f1 = 1.  # GP lengthscale
var_f2 = 1.  # GP variance
len_f2 = 1.  # GP lengthscale

kern1 = bayesnewton.kernels.Matern32(variance=var_f1, lengthscale=len_f1, fix_variance=True, fix_lengthscale=True)
kern2 = bayesnewton.kernels.Matern32(variance=var_f2, lengthscale=len_f2, fix_variance=True, fix_lengthscale=True)
kern = bayesnewton.kernels.Independent([kern1, kern2])
lik = bayesnewton.likelihoods.HeteroscedasticNoise()

if inf_method == 0:  # Newton
    if approx_method == 0:
        model = bayesnewton.models.MarkovNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
    elif approx_method == 1:
        model = bayesnewton.models.MarkovGaussNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
    elif approx_method == 2:
        model = bayesnewton.models.MarkovQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
    elif approx_method == 3:
        model = bayesnewton.models.MarkovNewtonRiemannGP(kernel=kern, likelihood=lik, X=X, Y=Y)
if inf_method == 1:  # VI
    if approx_method == 0:
        model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=X, Y=Y)
    elif approx_method == 1:
        model = bayesnewton.models.MarkovVariationalGaussNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
    elif approx_method == 2:
        model = bayesnewton.models.MarkovVariationalQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
    elif approx_method == 3:
        model = bayesnewton.models.MarkovVariationalRiemannGP(kernel=kern, likelihood=lik, X=X, Y=Y)
elif inf_method == 2:  # PEP
    if approx_method == 0:
        model = bayesnewton.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=X, Y=Y, power=0.5)
    elif approx_method == 1:
        raise NotImplementedError
    elif approx_method == 2:
        model = bayesnewton.models.MarkovExpectationPropagationQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y, power=0.5)
    elif approx_method == 3:
        model = bayesnewton.models.MarkovExpectationPropagationRiemannGP(kernel=kern, likelihood=lik, X=X, Y=Y, power=0.5)
if inf_method == 3:  # PL
    if approx_method == 0:
        model = bayesnewton.models.MarkovPosteriorLinearisation2ndOrderGP(kernel=kern, likelihood=lik, X=X, Y=Y)
    elif approx_method == 1:
        model = bayesnewton.models.MarkovPosteriorLinearisation2ndOrderGaussNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
    elif approx_method == 2:
        model = bayesnewton.models.MarkovPosteriorLinearisation2ndOrderQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
    elif approx_method == 3:
        model = bayesnewton.models.MarkovPosteriorLinearisation2ndOrderRiemannGP(kernel=kern, likelihood=lik, X=X, Y=Y)
    elif approx_method == 4:
        model = bayesnewton.models.MarkovPosteriorLinearisationGP(kernel=kern, likelihood=lik, X=X, Y=Y)
    # elif approx_method == 5:
    #     model = bayesnewton.models.MarkovPosteriorLinearisationQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
if inf_method == 4:  # first-order VI
    model = bayesnewton.models.FirstOrderMarkovVariationalGP(kernel=kern, likelihood=lik, X=X, Y=Y)


lr_adam = 0.1
lr_newton = 0.3
iters = 500
opt_hypers = objax.optimizer.Adam(model.vars(), beta2=0.99)
energy = objax.GradValues(model.energy, model.vars())

damping = 0.5


@objax.Function.with_vars(model.vars() + opt_hypers.vars())
def train_op():
    if inf_method < 4:
        model.inference(lr=lr_newton, damping=damping)  # perform inference and update variational params
    dE, E = energy()  # compute energy and its gradients w.r.t. hypers
    if inf_method == 4:
        opt_hypers(lr_adam, dE)
    test_nlpd_ = model.negative_log_predictive_density(X=XT, Y=YT)
    return E, test_nlpd_


train_op = objax.Jit(train_op)

losses = np.zeros(iters//10) * np.nan
nlpds = np.zeros(iters//10) * np.nan

t0 = time.time()
for i in range(1, iters + 1):
    loss, test_nlpd = train_op()
    print('iter %2d, energy: %1.4f, nlpd: %1.4f' % (i, loss[0], test_nlpd))
    if np.mod(i-1, 10) == 0:
        losses[(i-1)//10] = loss[0]
        nlpds[(i-1)//10] = test_nlpd
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

# print(losses)
# print(nlpds)

with open("output/hsced_" + str(inf_method) + "_" + str(approx_method) + "_" + str(fold) + "_loss.txt", "wb") as fp:
    pickle.dump(losses, fp)
with open("output/hsced_" + str(inf_method) + "_" + str(approx_method) + "_" + str(fold) + "_nlpd.txt", "wb") as fp:
    pickle.dump(nlpds, fp)

if plot_final:
    t0 = time.time()
    posterior_mean, posterior_var = model.predict(X=x_plot)
    nlpd = model.negative_log_predictive_density(X=XT, Y=YT)
    t1 = time.time()
    print('prediction time: %2.2f secs' % (t1-t0))
    print('NLPD: %1.2f' % nlpd)

    x_pred = X_scaler.inverse_transform(x_plot)
    link = model.likelihood.link_fn
    lb = posterior_mean[:, 0] - np.sqrt(posterior_var[:, 0, 0] + link(posterior_mean[:, 1]) ** 2) * 1.96
    ub = posterior_mean[:, 0] + np.sqrt(posterior_var[:, 0, 0] + link(posterior_mean[:, 1]) ** 2) * 1.96
    post_mean = y_scaler.inverse_transform(posterior_mean[:, 0])
    lb = y_scaler.inverse_transform(lb)
    ub = y_scaler.inverse_transform(ub)

    print('plotting ...')
    plt.figure(1, figsize=(12, 5))
    plt.clf()
    plt.plot(X_scaler.inverse_transform(X), y_scaler.inverse_transform(Y), 'k.', label='train')
    plt.plot(X_scaler.inverse_transform(XT), y_scaler.inverse_transform(YT), 'r.', label='test')
    plt.plot(x_pred, post_mean, 'c', label='posterior mean')
    plt.fill_between(x_pred, lb, ub, color='c', alpha=0.05, label='95% confidence')
    plt.xlim(x_pred[0], x_pred[-1])
    if hasattr(model, 'Z'):
        plt.plot(X_scaler.inverse_transform(model.Z.value[:, 0]),
                 (np.min(lb)-5)*np.ones_like(model.Z.value[:, 0]),
                 'c^',
                 markersize=4)
    plt.legend()
    plt.title('Heteroscedastic Noise Model (motorcycle crash data)')
    plt.xlabel('time (milliseconds)')
    plt.ylabel('accelerometer reading')
    plt.show()
