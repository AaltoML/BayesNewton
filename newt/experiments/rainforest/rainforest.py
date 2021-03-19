import newt
import objax
import numpy as np
import time
import pickle
import sys

print('loading rainforest data ...')
data = np.loadtxt('../../data/TRI2TU-data.csv', delimiter=',')

spatial_points = np.array([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])

if len(sys.argv) > 1:
    model_type = int(sys.argv[1])
    nr_ind = int(sys.argv[2])
    fold = int(sys.argv[3])
else:
    model_type = 0
    nr_ind = 1
    nr = 100  # spatial grid point (y-axis)
    fold = 0

nr = spatial_points[nr_ind]
nt = 200  # temporal grid points (x-axis)
scale = 1000 / nt

t, r, Y_ = newt.utils.discretegrid(data, [0, 1000, 0, 500], [nt, nr])
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
t_test, r_test, Y_test = newt.utils.create_spatiotemporal_grid(X_test, Y_test)

var_f = 1.  # GP variance
len_f = 10.  # lengthscale

kern = newt.kernels.SpatialMatern32(variance=var_f, lengthscale=len_f, z=r[0, ...], sparse=False)
lik = newt.likelihoods.Poisson()
if model_type == 0:
    model = newt.models.MarkovGP(kernel=kern, likelihood=lik, X=t, R=r, Y=Y)
elif model_type == 1:
    model = newt.models.MarkovGPMeanField(kernel=kern, likelihood=lik, X=t, R=r, Y=Y)
elif model_type == 2:
    model = newt.models.InfiniteHorizonGP(kernel=kern, likelihood=lik, X=t, R=r, Y=Y)

print('num spatial pts:', nr)
print(model)

inf = newt.inference.VariationalInference(cubature=newt.cubature.Unscented())

trainable_vars = model.vars() + inf.vars()
energy = objax.GradValues(inf.energy, trainable_vars)

lr_adam = 0.2
lr_newton = 0.2
iters = 100
opt = objax.optimizer.Adam(trainable_vars)


def train_op():
    inf(model, lr=lr_newton)  # perform inference and update variational params
    dE, E = energy(model)  # compute energy and its gradients w.r.t. hypers
    return dE, E


train_op = objax.Jit(train_op, trainable_vars)

t0 = time.time()
for i in range(1, iters + 1):
    grad, loss = train_op()
    opt(lr_adam, grad)
    print('iter %2d: energy: %1.4f' % (i, loss[0]))
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

# calculate posterior predictive distribution via filtering and smoothing at train & test locations:
print('calculating the posterior predictive distribution ...')
t0 = time.time()
nlpd = model.negative_log_predictive_density(X=t_test, R=r_test, Y=Y_test)
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))
print('nlpd: %2.3f' % nlpd)

with open("output/" + str(model_type) + "_" + str(nr_ind) + "_" + str(fold) + "_nlpd.txt", "wb") as fp:
    pickle.dump(nlpd, fp)
