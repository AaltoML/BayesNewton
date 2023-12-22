import bayesnewton
import objax
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler

print('loading data ...')
D = np.loadtxt('../data/mcycle.csv', delimiter=',')
X = D[:, 1:2]
Y = D[:, 2:]

# Standardize
X_scaler = StandardScaler().fit(X)
y_scaler = StandardScaler().fit(Y)
Xall = X_scaler.transform(X)
Yall = y_scaler.transform(Y)
x_plot = np.linspace(np.min(Xall)-0.2, np.max(Xall)+0.2, 200)[:, None]

# Load cross-validation indices
cvind = np.loadtxt('../experiments/motorcycle/cvind.csv').astype(int)

# 10-fold cross-validation setup
nt = np.floor(cvind.shape[0]/10).astype(int)
cvind = np.reshape(cvind[:10*nt], (10, nt))

np.random.seed(123)
fold = 3

# Get training and test indices
test = cvind[fold, :]
train = np.setdiff1d(cvind, test)

# Set training and test data
X = Xall[train, :]
Y = Yall[train, :]
XT = Xall[test, :]
YT = Yall[test, :]
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

# --- GP ---
# model = bayesnewton.models.NewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model = bayesnewton.models.VariationalGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model = bayesnewton.models.ExpectationPropagationGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model = bayesnewton.models.VariationalQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# --- sparse ---
# model = bayesnewton.models.SparseVariationalGP(kernel=kern, likelihood=lik, X=X, Y=Y, Z=Z)
# model = bayesnewton.models.SparseExpectationPropagationGP(kernel=kern, likelihood=lik, X=X, Y=Y, Z=Z)
# --- sparse quasi-Newton ---
# model = bayesnewton.models.SparseQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y, Z=Z)
# model = bayesnewton.models.SparseVariationalQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y, Z=Z)
# model = bayesnewton.models.SparseExpectationPropagationQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y, Z=Z, power=0.5)

lr_adam = 0.01
lr_newton = 0.3
iters = 300
opt_hypers = objax.optimizer.Adam(model.vars())
energy = objax.GradValues(model.energy, model.vars())

damping = 0.5


@objax.Function.with_vars(model.vars() + opt_hypers.vars())
def train_op():
    model.inference(lr=lr_newton, damping=damping)  # perform inference and update variational params
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

print('avg cross cov:', np.mean(np.abs(posterior_var[:, 0, 1])))

x_pred = X_scaler.inverse_transform(x_plot)
link = model.likelihood.link_fn
lb = posterior_mean[:, 0] - np.sqrt(posterior_var[:, 0, 0] + link(posterior_mean[:, 1]) ** 2) * 1.96
ub = posterior_mean[:, 0] + np.sqrt(posterior_var[:, 0, 0] + link(posterior_mean[:, 1]) ** 2) * 1.96
post_mean = y_scaler.inverse_transform(posterior_mean[:, 0:1])
lb = y_scaler.inverse_transform(lb[:, None])[:, 0]
ub = y_scaler.inverse_transform(ub[:, None])[:, 0]

print('plotting ...')
plt.figure(1, figsize=(12, 5))
plt.clf()
plt.plot(X_scaler.inverse_transform(X), y_scaler.inverse_transform(Y), 'k.', label='train')
plt.plot(X_scaler.inverse_transform(XT), y_scaler.inverse_transform(YT), 'r.', label='test')
plt.plot(x_pred, post_mean, 'c', label='posterior mean')
plt.fill_between(x_pred[:, 0], lb, ub, color='c', alpha=0.05, label='95% confidence')
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
