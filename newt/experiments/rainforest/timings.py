import newt
import objax
import numpy as np
import time
import pickle
import sys

data = np.loadtxt('../../data/TRI2TU-data.csv', delimiter=',')

spatial_points = np.array([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])

if len(sys.argv) > 1:
    model_type = int(sys.argv[1])
    nr_ind = int(sys.argv[2])
else:
    model_type = 0
    nr_ind = 1
    nr = 100  # spatial grid point (y-axis)
nr = spatial_points[nr_ind]
nt = 200  # temporal grid points (x-axis)
scale = 1000 / nt

t, r, Y_ = newt.utils.discretegrid(data, [0, 1000, 0, 500], [nt, nr])

np.random.seed(99)
N = nr * nt  # number of data points

test_ind = np.random.permutation(N)[:N//10]  # [:N//4]
Y = Y_.flatten()
Y[test_ind] = np.nan
Y = Y.reshape(nt, nr)

var_f = 1.  # GP variance
len_f = 10.  # lengthscale

kern = newt.kernels.SpatialMatern32(variance=var_f, lengthscale=len_f, z=r[0, ...], sparse=False)
lik = newt.likelihoods.Poisson()
# lik = newt.likelihoods.Gaussian()
# model = newt.models.GP(kernel=kern, likelihood=lik, X=x, Y=Y)
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
lr_newton = 0.1
iters = 11
opt = objax.optimizer.Adam(trainable_vars)


def train_op():
    inf(model, lr=lr_newton)  # perform inference and update variational params
    dE, E = energy(model)  # compute energy and its gradients w.r.t. hypers
    return dE, E


train_op = objax.Jit(train_op, trainable_vars)

for i in range(1, iters + 1):
    if i == 2:
        t0 = time.time()
    grad, loss = train_op()
    opt(lr_adam, grad)
    print('iter %2d: energy: %1.4f' % (i, loss[0]))
t1 = time.time()
avg_time_taken = (t1-t0)/(iters - 1)
print('optimisation time: %2.2f secs' % avg_time_taken)

with open("output/" + str(model_type) + "_" + str(nr_ind) + "_time.txt", "wb") as fp:
    pickle.dump(avg_time_taken, fp)
