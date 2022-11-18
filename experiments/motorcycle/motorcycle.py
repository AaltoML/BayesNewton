import bayesnewton
import objax
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from sklearn.preprocessing import StandardScaler
import sys

plot_intermediate = False

print('loading data ...')
D = np.loadtxt('../../data/mcycle.csv', delimiter=',')
X = D[:, 1:2]
Y = D[:, 2:]
N_batch = 100
M = 30

# Standardize
X_scaler = StandardScaler().fit(X)
y_scaler = StandardScaler().fit(Y)
Xall = X_scaler.transform(X)
Yall = y_scaler.transform(Y)
Z = np.linspace(np.min(Xall), np.max(Xall), M)
x_plot = np.linspace(np.min(Xall)-0.2, np.max(Xall)+0.2, 200)

# Load cross-validation indices
cvind = np.loadtxt('cvind.csv').astype(int)

# 10-fold cross-validation setup
nt = np.floor(cvind.shape[0]/10).astype(int)
cvind = np.reshape(cvind[:10*nt], (10, nt))

np.random.seed(123)

if len(sys.argv) > 1:
    method = int(sys.argv[1])
    fold = int(sys.argv[2])
    plot_final = False
else:
    method = 4
    fold = 8
    plot_final = True

if len(sys.argv) > 3:
    baseline = bool(int(sys.argv[3]))
else:
    baseline = True

if len(sys.argv) > 4:
    parallel = bool(int(sys.argv[4]))
else:
    parallel = None

print('method number:', method)
print('batch number:', fold)
print('baseline:', baseline)
print('parallel:', parallel)

# Get training and test indices
test = cvind[fold, :]
train = np.setdiff1d(cvind, test)

# Set training and test data
X = Xall[train, :]
Y = Yall[train, :]
XT = Xall[test, :]
YT = Yall[test, :]
N = X.shape[0]

var_f1 = 3.  # GP variance
len_f1 = 1.  # GP lengthscale
var_f2 = 3.  # GP variance
len_f2 = 1.  # GP lengthscale

kern1 = bayesnewton.kernels.Matern32(variance=var_f1, lengthscale=len_f1)
kern2 = bayesnewton.kernels.Matern32(variance=var_f2, lengthscale=len_f2)
kern = bayesnewton.kernels.Independent([kern1, kern2])
lik = bayesnewton.likelihoods.HeteroscedasticNoise()

if baseline:
    if method == 0:
        model = bayesnewton.models.MarkovTaylorGP(kernel=kern, likelihood=lik, X=X, Y=Y, parallel=parallel)
    elif method == 1:
        model = bayesnewton.models.MarkovPosteriorLinearisationGP(kernel=kern, likelihood=lik, X=X, Y=Y,
                                                                  parallel=parallel)
    elif method == 2:
        model = bayesnewton.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=X, Y=Y,
                                                       parallel=parallel, power=1.)
    elif method == 3:
        model = bayesnewton.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=X, Y=Y,
                                                       parallel=parallel, power=0.5)
    elif method == 4:
        model = bayesnewton.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=X, Y=Y,
                                                                  parallel=parallel, power=0.01)
    elif method == 4:
        model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=X, Y=Y,
                                                                  parallel=parallel)
else:
    if method == 0:
        model = bayesnewton.models.SparseMarkovTaylorGP(kernel=kern, likelihood=lik, X=X, Y=Y, Z=Z, parallel=parallel)
    elif method == 1:
        model = bayesnewton.models.SparseMarkovPosteriorLinearisationGP(kernel=kern, likelihood=lik, X=X, Y=Y, Z=Z,
                                                                  parallel=parallel)
    elif method == 2:
        model = bayesnewton.models.SparseMarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=X, Y=Y, Z=Z,
                                                       parallel=parallel, power=1.)
    elif method == 3:
        model = bayesnewton.models.SparseMarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=X, Y=Y, Z=Z,
                                                       parallel=parallel, power=0.5)
    elif method == 4:
        model = bayesnewton.models.SparseMarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=X, Y=Y, Z=Z,
                                                                  parallel=parallel, power=0.01)
    elif method == 4:
        model = bayesnewton.models.SparseMarkovVariationalGP(kernel=kern, likelihood=lik, X=X, Y=Y, Z=Z,
                                                                  parallel=parallel)


lr_adam = 0.025
# lr_adam = 0.01
lr_newton = .05
# lr_newton = 0.01
iters = 500
opt_hypers = objax.optimizer.Adam(model.vars())
energy = objax.GradValues(model.energy, model.vars())


@objax.Function.with_vars(model.vars() + opt_hypers.vars())
def train_op():
    model.inference(lr=lr_newton)  # perform inference and update variational params
    dE, E = energy()  # compute energy and its gradients w.r.t. hypers
    opt_hypers(lr_adam, dE)
    return E


train_op = objax.Jit(train_op)


print('optimising the hyperparameters ...')
t0 = time.time()
for i in range(1, iters + 1):
    loss = train_op()
    print('iter %2d, energy: %1.4f' % (i, loss[0]))
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

# calculate posterior predictive distribution via filtering and smoothing at train & test locations:
print('calculating the posterior predictive distribution ...')
t0 = time.time()
posterior_mean, posterior_var = model.predict(X=x_plot)
nlpd = model.negative_log_predictive_density(X=XT, Y=YT)
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))
print('NLPD: %1.2f' % nlpd)

if baseline:
    with open("output/baseline_" + str(method) + "_" + str(fold) + "_nlpd.txt", "wb") as fp:
        pickle.dump(nlpd, fp)
else:
    with open("output/" + str(method) + "_" + str(fold) + "_nlpd.txt", "wb") as fp:
        pickle.dump(nlpd, fp)

# with open("output/" + str(method) + "_" + str(fold) + "_nlpd.txt", "rb") as fp:
#     nlpd_show = pickle.load(fp)
# print(nlpd_show)

if plot_final:
    x_pred = X_scaler.inverse_transform(x_plot)
    link = model.likelihood.link_fn
    lb = posterior_mean[:, 0] - np.sqrt(posterior_var[:, 0] + link(posterior_mean[:, 1]) ** 2) * 1.96
    ub = posterior_mean[:, 0] + np.sqrt(posterior_var[:, 0] + link(posterior_mean[:, 1]) ** 2) * 1.96
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
                 (np.min(lb) - 5) * np.ones_like(model.Z.value[:, 0]),
                 'c^',
                 markersize=4)
    plt.legend()
    plt.title('Heteroscedastic Noise Model via Kalman smoothing (motorcycle crash data)')
    plt.xlabel('time (milliseconds)')
    plt.ylabel('accelerometer reading')
    plt.show()
