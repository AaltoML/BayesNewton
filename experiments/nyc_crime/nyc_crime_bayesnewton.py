import bayesnewton
import objax
import numpy as np
import pickle
import time
import sys
from scipy.cluster.vq import kmeans2
from jax.lib import xla_bridge

if len(sys.argv) > 1:
    ind = int(sys.argv[1])
else:
    ind = 0


if len(sys.argv) > 2:
    mean_field = bool(int(sys.argv[2]))
else:
    mean_field = False


if len(sys.argv) > 3:
    parallel = bool(int(sys.argv[3]))
else:
    parallel = None

# ===========================Load Data===========================
train_data = pickle.load(open("data/train_data_" + str(ind) + ".pickle", "rb"))
pred_data = pickle.load(open("data/pred_data_" + str(ind) + ".pickle", "rb"))

X = train_data['X']
Y = train_data['Y']

X_t = pred_data['test']['X']
Y_t = pred_data['test']['Y']

bin_sizes = train_data['bin_sizes']
binsize = np.prod(bin_sizes)

print('X: ', X.shape)

num_z_space = 30

grid = True
print(Y.shape)
print("num data points =", Y.shape[0])

if grid:
    # the gridded approach:
    t, R, Y = bayesnewton.utils.create_spatiotemporal_grid(X, Y)
    t_t, R_t, Y_t = bayesnewton.utils.create_spatiotemporal_grid(X_t, Y_t)
else:
    # the sequential approach:
    t = X[:, :1]
    R = X[:, 1:]
    t_t = X_t[:, :1]
    R_t = X_t[:, 1:]
Nt = t.shape[0]
print("num time steps =", Nt)
N = Y.shape[0] * Y.shape[1] * Y.shape[2]
print("num data points =", N)

var_f = 1.
len_time = 0.001
len_space = 0.1

sparse = True
opt_z = True  # will be set to False if sparse=False

if sparse:
    z = kmeans2(R[0, ...], num_z_space, minit="points")[0]
else:
    z = R[0, ...]

kern_time = bayesnewton.kernels.Matern32(variance=var_f, lengthscale=len_time)
kern_space0 = bayesnewton.kernels.Matern32(variance=var_f, lengthscale=len_space)
kern_space1 = bayesnewton.kernels.Matern32(variance=var_f, lengthscale=len_space)
kern_space = bayesnewton.kernels.Separable([kern_space0, kern_space1])

kern = bayesnewton.kernels.SpatioTemporalKernel(temporal_kernel=kern_time,
                                         spatial_kernel=kern_space,
                                         z=z,
                                         sparse=sparse,
                                         opt_z=opt_z,
                                         conditional='Full')

lik = bayesnewton.likelihoods.Poisson(binsize=binsize)

if mean_field:
    model = bayesnewton.models.MarkovVariationalMeanFieldGP(kernel=kern, likelihood=lik, X=t, R=R, Y=Y, parallel=parallel)
else:
    model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=t, R=R, Y=Y, parallel=parallel)

lr_adam = 0.01
lr_newton = 0.1
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

t0 = time.time()
for i in range(1, iters):
    loss = train_op()
    print('iter %2d: energy: %1.4f' % (i, loss[0]))
t1 = time.time()
# print('optimisation time: %2.2f secs' % (t1-t0))
avg_time_taken = (t1-t0)/iters
print('average iter time: %2.2f secs' % avg_time_taken)

posterior_mean, posterior_var = model.predict_y(X=t_t, R=R_t)
nlpd = model.negative_log_predictive_density(X=t_t, R=R_t, Y=Y_t)
rmse = np.sqrt(np.nanmean((np.squeeze(Y_t) - np.squeeze(posterior_mean))**2))
print('nlpd: %2.3f' % nlpd)
print('rmse: %2.3f' % rmse)

cpugpu = xla_bridge.get_backend().platform

with open("output/" + str(int(mean_field)) + "_" + str(ind) + "_" + str(int(parallel)) + "_" + cpugpu + "_time.txt", "wb") as fp:
    pickle.dump(avg_time_taken, fp)
with open("output/" + str(int(mean_field)) + "_" + str(ind) + "_" + str(int(parallel)) + "_" + cpugpu + "_nlpd.txt", "wb") as fp:
    pickle.dump(nlpd, fp)
with open("output/" + str(int(mean_field)) + "_" + str(ind) + "_" + str(int(parallel)) + "_" + cpugpu + "_rmse.txt", "wb") as fp:
    pickle.dump(rmse, fp)
