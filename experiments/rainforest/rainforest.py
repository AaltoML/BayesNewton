import bayesnewton
import objax
import numpy as np
import time
import pickle
import sys

print('loading rainforest data ...')
data = np.loadtxt('../../data/TRI2TU-data.csv', delimiter=',')

spatial_points = np.array([20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])

if len(sys.argv) > 1:
    model_type = int(sys.argv[1])
    nr_ind = int(sys.argv[2])
    fold = int(sys.argv[3])
    parallel = bool(int(sys.argv[4]))
else:
    model_type = 0
    nr_ind = 1
    # nr = 100  # spatial grid points (y-axis)
    fold = 0
    parallel = None

nr = spatial_points[nr_ind]  # spatial grid points (y-axis)
nt = 200  # temporal grid points (x-axis)
scale = 1000 / nt

t, r, Y_ = bayesnewton.utils.discretegrid(data, [0, 1000, 0, 500], [nt, nr])
t_flat, r_flat, Y_flat = t.flatten(), r.flatten(), Y_.flatten()

N = nr * nt  # number of data points

# sort out the train/test split
np.random.seed(99)
ind_shuffled = np.random.permutation(N)
ind_split = np.stack(np.split(ind_shuffled, 10))  # 10 random batches of data indices
test_ind = ind_split[fold]  # test_ind = np.random.permutation(N)[:N//10]
t_test = t_flat[test_ind]
r_test = r_flat[test_ind]
Y_test = Y_flat[test_ind]
Y_flat[test_ind] = np.nan
Y = Y_flat.reshape(nt, nr)

# put test points on a grid to speed up prediction
X_test = np.concatenate([t_test[:, None], r_test[:, None]], axis=1)
t_test, r_test, Y_test = bayesnewton.utils.create_spatiotemporal_grid(X_test, Y_test)

var_f = 1.  # GP variance
len_f = 10.  # lengthscale

kern = bayesnewton.kernels.SpatialMatern32(variance=var_f, lengthscale=len_f, z=r[0, ...], sparse=False)
lik = bayesnewton.likelihoods.Poisson()
if model_type == 0:
    model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=t, R=r, Y=Y, parallel=parallel)
elif model_type == 1:
    model = bayesnewton.models.MarkovVariationalMeanFieldGP(kernel=kern, likelihood=lik, X=t, R=r, Y=Y, parallel=parallel)
elif model_type == 2:
    model = bayesnewton.models.InfiniteHorizonVariationalGP(kernel=kern, likelihood=lik, X=t, R=r, Y=Y, parallel=parallel)

print('num spatial pts:', nr)
print('batch number:', fold)
print('parallel:', parallel)
print(model)

lr_adam = 0.2
lr_newton = 0.2
iters = 11
opt_hypers = objax.optimizer.Adam(model.vars())
energy = objax.GradValues(model.energy, model.vars())

unscented_transform = bayesnewton.cubature.Unscented(dim=1)  # 5th-order unscented transform


@objax.Function.with_vars(model.vars() + opt_hypers.vars())
def train_op():
    model.inference(lr=lr_newton, cubature=unscented_transform)  # perform inference and update variational params
    dE, E = energy(cubature=unscented_transform)  # compute energy and its gradients w.r.t. hypers
    opt_hypers(lr_adam, dE)
    return E


train_op = objax.Jit(train_op)

t0 = time.time()
for i in range(1, iters + 1):
    if i == 2:
        t0 = time.time()
    loss = train_op()
    print('iter %2d: energy: %1.4f' % (i, loss[0]))
t1 = time.time()
# print('optimisation time: %2.2f secs' % (t1-t0))
avg_time_taken = (t1-t0)/(iters - 1)
print('average iter time: %2.2f secs' % avg_time_taken)

with open("output/" + str(model_type) + "_" + str(nr_ind) + "_" + str(int(parallel)) + "_time.txt", "wb") as fp:
    pickle.dump(avg_time_taken, fp)

# calculate posterior predictive distribution via filtering and smoothing at train & test locations:
print('calculating the posterior predictive distribution ...')
t0 = time.time()
nlpd = model.negative_log_predictive_density(X=t_test, R=r_test, Y=Y_test)
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))
print('nlpd: %2.3f' % nlpd)
#
# with open("output/" + str(model_type) + "_" + str(nr_ind) + "_" + str(fold) + "_nlpd.txt", "wb") as fp:
#     pickle.dump(nlpd, fp)
