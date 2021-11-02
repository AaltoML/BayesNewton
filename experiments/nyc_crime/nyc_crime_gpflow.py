import tensorflow as tf
import gpflow
from gpflow.optimizers import NaturalGradient
from gpflow.utilities import set_trainable, leaf_components
import numpy as np
import scipy as sp
import time
from scipy.cluster.vq import kmeans2
from tqdm import tqdm
import pickle
import sys

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


if len(sys.argv) > 1:
    ind = int(sys.argv[1])
else:
    ind = 0


if len(sys.argv) > 2:
    num_z_ind = int(sys.argv[2])
else:
    num_z_ind = 0


# ===========================Load Data===========================
train_data = pickle.load(open("data/train_data_" + str(ind) + ".pickle", "rb"))
pred_data = pickle.load(open("data/pred_data_" + str(ind) + ".pickle", "rb"))

X = train_data['X']
Y = train_data['Y']

X_t = pred_data['test']['X']
Y_t = pred_data['test']['Y']

bin_sizes = train_data['bin_sizes']
binsize = np.prod(bin_sizes)

non_nan_idx = np.logical_not(np.isnan(np.squeeze(Y)))
X = X[non_nan_idx, :]
Y = Y[non_nan_idx, :]

non_nan_idx_t = np.logical_not(np.isnan(np.squeeze(Y_t)))
X_t = X_t[non_nan_idx_t, :]
Y_t = Y_t[non_nan_idx_t, :]

print('X: ', X.shape)

kernel_lengthscales = [0.001, 0.1, 0.1]
kernel_variances = 1.0
train_z = True
epochs = 500
step_size = 0.01
# jitter = 1e-4
natgrad_step_size = 0.1
# enforce_psd = False
minibatch_size = [1500, 3000]
num_z = [1500, 3000]


def get_gpflow_params(m):
    params = {}
    leafs = leaf_components(m)
    for key in leafs.keys():
        tf_vars = leafs[key].trainable_variables

        # check if variable exists
        if len(tf_vars) == 1:
            tf_var = tf_vars[0]

            params[key] = tf_var.numpy()

    return params


N, D = X.shape

print('num_z: ', num_z[num_z_ind])
Z_all = kmeans2(X, num_z[num_z_ind], minit="points")[0]

kernel = gpflow.kernels.Matern32

k = None
for d in range(D):
    # print(d, kernel_lengthscales)
    if type(kernel_lengthscales) is list:
        k_ls = kernel_lengthscales[d]
    else:
        k_ls = kernel_lengthscales

    if type(kernel_variances) is list:
        k_var = kernel_variances[d]
    else:
        k_var = kernel_variances

    k_d = kernel(
        lengthscales=[k_ls],
        variance=k_var,
        active_dims=[d]
    )

    # print(k_d)
    if k is None:
        k = k_d
    else:
        k = k * k_d

init_as_cvi = True

if init_as_cvi:
    M = Z_all.shape[0]
    jit = 1e-6

    Kzz = k(Z_all, Z_all)

    def inv(K):
        K_chol = sp.linalg.cholesky(K + jit * np.eye(M), lower=True)
        return sp.linalg.cho_solve((K_chol, True), np.eye(K.shape[0]))

    # manual q(u) decompositin
    nat1 = np.zeros([M, 1])
    nat2 = -0.5 * inv(Kzz)

    lam1 = 1e-5 * np.ones([M, 1])
    lam2 = -0.5 * np.eye(M)

    S = inv(-2 * (nat2 + lam2))
    m = S @ (lam1 + nat1)

    S_chol = sp.linalg.cholesky(S + jit * np.eye(M), lower=True)
    S_flattened = S_chol[np.tril_indices(M, 0)]

    q_mu = m
    q_sqrt = np.array([S_chol])
else:
    q_mu = 1e-5 * np.ones([Z_all.shape[0], 1])  # match gpjax init
    q_sqrt = None

lik = gpflow.likelihoods.Poisson(binsize=binsize)

data = (X, Y)

m = gpflow.models.SVGP(
    inducing_variable=Z_all,
    whiten=True,
    kernel=k,
    mean_function=None,
    likelihood=lik,
    q_mu=q_mu,
    q_sqrt=q_sqrt
)

set_trainable(m.inducing_variable, True)

# ===========================Train===========================

if minibatch_size[num_z_ind] is None or minibatch_size[num_z_ind] is 'none':
    training_loss = m.training_loss_closure(
        data
    )
else:
    print(N, minibatch_size[num_z_ind])
    train_dataset = (tf.data.Dataset.from_tensor_slices(data).repeat().shuffle(N).batch(minibatch_size[num_z_ind]))
    train_iter = iter(train_dataset)
    training_loss = m.training_loss_closure(train_iter)


# make it so adam does not train these
set_trainable(m.q_mu, False)
set_trainable(m.q_sqrt, False)

natgrad_opt = NaturalGradient(gamma=natgrad_step_size)
variational_params = [(m.q_mu, m.q_sqrt)]

optimizer = tf.optimizers.Adam

adam_opt_for_vgp = optimizer(step_size)

loss_arr = []

bar = tqdm(total=epochs)

# MINIBATCHING TRAINING
t0 = time.time()
for i in range(epochs):
    # NAT GRAD STEP
    natgrad_opt.minimize(training_loss, var_list=variational_params)

    # elbo = -m.elbo(data).numpy()

    # loss_arr.append(elbo)

    # ADAM STEP
    adam_opt_for_vgp.minimize(training_loss, var_list=m.trainable_variables)

    bar.update(1)
t1 = time.time()
avg_time_taken = (t1-t0)/epochs
print('average iter time: %2.2f secs' % avg_time_taken)


def _prediction_fn(X_, Y_):
    mu, var = m.predict_y(X_)
    log_pred_density = m.predict_log_density((X_, Y_))
    return mu.numpy(), var.numpy(), log_pred_density.numpy()


print('predicting...')
posterior_mean, posterior_var, lpd = _prediction_fn(X_t, Y_t)
# print(lpd.shape)
# print(lpd)
nlpd = np.mean(-lpd)
rmse = np.sqrt(np.nanmean((np.squeeze(Y_t) - np.squeeze(posterior_mean))**2))
print('nlpd: %2.3f' % nlpd)
print('rmse: %2.3f' % rmse)

# prediction_fn = lambda X: utils.batch_predict(X, _prediction_fn, verbose=True)

if len(tf.config.list_physical_devices('GPU')) > 0:
    cpugpu = 'gpu'
else:
    cpugpu = 'cpu'

with open("output/gpflow_" + str(ind) + "_" + str(num_z_ind) + "_" + cpugpu + "_time.txt", "wb") as fp:
    pickle.dump(avg_time_taken, fp)
with open("output/gpflow_" + str(ind) + "_" + str(num_z_ind) + "_" + cpugpu + "_nlpd.txt", "wb") as fp:
    pickle.dump(nlpd, fp)
with open("output/gpflow_" + str(ind) + "_" + str(num_z_ind) + "_" + cpugpu + "_rmse.txt", "wb") as fp:
    pickle.dump(rmse, fp)
