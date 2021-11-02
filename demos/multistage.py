import bayesnewton
import objax
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import gpflow
from markovflow.likelihoods.mutlistage_likelihood import MultiStageLikelihood

N = 100  # number of training points
X_train = np.arange(N).astype(float)
L = 3  # number of latent functions

# Define the kernel
k1a = gpflow.kernels.Periodic(
    gpflow.kernels.Matern52(variance=1.0, lengthscales=3.0), period=12.0
)
k1b = gpflow.kernels.Matern52(variance=1.0, lengthscales=30.0)
k2 = gpflow.kernels.Matern32(variance=0.1, lengthscales=5.0)
k = k1a * k1b + k2

# Draw three independent functions from the same Gaussian process
X = X_train
num_latent = L
K = k(X[:, None])
np.random.seed(123)
v = np.random.randn(len(K), num_latent)
# We draw samples from a GP with kernel k(.) evaluated at X by reparameterizing:
# f ~ N(0, K) → f = chol(K) v, v ~ N(0, I), where chol(K) chol(K)ᵀ = K
f = np.linalg.cholesky(K + 1e-6 * np.eye(len(K))) @ v

# We shift the third function to increase the mean of the Poisson component to 20 to make it easier to identify
f += np.array([0.0, 0.0, np.log(20)]).reshape(1, L)

# Define the likelihood
lik = MultiStageLikelihood()
# Draw observations from the likelihood given the functions `f` from the previous step
Y = lik.sample_y(tf.convert_to_tensor(f, dtype=gpflow.default_float())).numpy()

# Plot all three functions
# plt.figure(1)
# for i in range(num_latent):
#     plt.plot(X, f[:, i])
# _ = plt.xticks(np.arange(0, 100, 12))
# # Plot the observations
# plt.figure(2)
# _ = plt.plot(X, Y, ".")
# plt.show()

var_f = 1.0  # GP variance
len_f = 15.0  # GP lengthscale

kern1 = bayesnewton.kernels.Matern52(variance=var_f, lengthscale=len_f)
kern2 = bayesnewton.kernels.Matern52(variance=var_f, lengthscale=len_f)
kern3 = bayesnewton.kernels.Matern52(variance=var_f, lengthscale=len_f)
kern = bayesnewton.kernels.Independent(kernels=[kern1, kern2, kern3])
lik = bayesnewton.likelihoods.MultiStage()

# model = bayesnewton.models.MarkovNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model = bayesnewton.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=X, Y=Y, power=0.5)
# -- Gauss-Newton ---
# model = bayesnewton.models.MarkovGaussNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model = bayesnewton.models.MarkovVariationalGaussNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# -- quasi-Newton ---
# model = bayesnewton.models.MarkovQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model = bayesnewton.models.MarkovVariationalQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
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


@objax.Function.with_vars(model.vars() + opt_hypers.vars())
def train_op():
    model.inference(lr=lr_newton, damping=damping, cubature=unscented_transform, ensure_psd=False)  # perform inference and update variational params
    dE, E = energy(cubature=unscented_transform)  # compute energy and its gradients w.r.t. hypers
    # opt_hypers(lr_adam, dE)
    test_nlpd_ = 0.  # model.negative_log_predictive_density(X=XT, Y=YT, cubature=unscented_transform)
    return E, test_nlpd_


train_op = objax.Jit(train_op)

t0 = time.time()
for i in range(1, iters + 1):
    loss, test_nlpd = train_op()
    print('iter %2d, energy: %1.4f, nlpd: %1.4f' % (i, loss[0], test_nlpd))
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))
