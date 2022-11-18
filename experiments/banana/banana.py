import sys
import bayesnewton
import objax
import numpy as np
import time
import pickle

print('loading banana data ...')
np.random.seed(99)
inputs = np.loadtxt('../../data/banana_large.csv', delimiter=',', skiprows=1)
Xall = inputs[:, :1]  # temporal inputs (x-axis)
Rall = inputs[:, 1:2]  # spatial inputs (y-axis)
Yall = np.maximum(inputs[:, 2:], 0)  # observations / labels

N = Xall.shape[0]  # number of training points
M = 15
Z = np.linspace(-3., 3., M)  # inducing points

ind_shuffled = np.random.permutation(N)
ind_split = np.stack(np.split(ind_shuffled, 10))  # 10 random batches of data indices

# Test points
Xtest, Rtest = np.mgrid[-3.2:3.2:100j, -3.2:3.2:100j]

if len(sys.argv) > 1:
    method = int(sys.argv[1])
    fold = int(sys.argv[2])
else:
    method = 3
    fold = 0

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
ind_test = ind_split[fold]  # np.sort(ind_shuffled[:N//10])
ind_train = np.concatenate(ind_split[np.arange(10) != fold])

# Set training and test data
X = Xall[ind_train]
R = Rall[ind_train]
Y = Yall[ind_train]
XT = Xall[ind_test]
RT = Rall[ind_test]
YT = Yall[ind_test]

var_f = 1.  # GP variance
len_time = 1.  # temporal lengthscale
len_space = 1.  # spacial lengthscale

kern = bayesnewton.kernels.SpatioTemporalMatern52(variance=var_f, lengthscale_time=len_time, lengthscale_space=len_space,
                                           z=np.linspace(-3, 3, M), sparse=True, opt_z=False, conditional='Full')
lik = bayesnewton.likelihoods.Bernoulli(link='logit')


if baseline:
    if method == 0:
        model = bayesnewton.models.MarkovTaylorGP(kernel=kern, likelihood=lik, X=X, R=R, Y=Y, parallel=parallel)
    elif method == 1:
        model = bayesnewton.models.MarkovPosteriorLinearisationGP(kernel=kern, likelihood=lik, X=X, R=R, Y=Y,
                                                                  parallel=parallel)
    elif method == 2:
        model = bayesnewton.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=X, R=R, Y=Y,
                                                       parallel=parallel, power=1.)
    elif method == 3:
        model = bayesnewton.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=X, R=R, Y=Y,
                                                       parallel=parallel, power=0.5)
    elif method == 4:
        model = bayesnewton.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=X, R=R, Y=Y,
                                                                  parallel=parallel, power=0.01)
    elif method == 4:
        model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=X, R=R, Y=Y,
                                                                  parallel=parallel)
else:
    if method == 0:
        model = bayesnewton.models.SparseMarkovTaylorGP(kernel=kern, likelihood=lik, X=X, R=R, Y=Y, Z=Z, parallel=parallel)
    elif method == 1:
        model = bayesnewton.models.SparseMarkovPosteriorLinearisationGP(kernel=kern, likelihood=lik, X=X, R=R, Y=Y, Z=Z,
                                                                  parallel=parallel)
    elif method == 2:
        model = bayesnewton.models.SparseMarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=X, R=R, Y=Y, Z=Z,
                                                       parallel=parallel, power=1.)
    elif method == 3:
        model = bayesnewton.models.SparseMarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=X, R=R, Y=Y, Z=Z,
                                                       parallel=parallel, power=0.5)
    elif method == 4:
        model = bayesnewton.models.SparseMarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=X, R=R, Y=Y, Z=Z,
                                                                  parallel=parallel, power=0.01)
    elif method == 4:
        model = bayesnewton.models.SparseMarkovVariationalGP(kernel=kern, likelihood=lik, X=X, R=R, Y=Y, Z=Z,
                                                                  parallel=parallel)


lr_adam = 0.1
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
for i in range(1, iters + 1):
    loss = train_op()
    print('iter %2d, energy: %1.4f' % (i, loss[0]))
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

print('calculating the posterior predictive distribution ...')
t0 = time.time()
nlpd = model.negative_log_predictive_density(X=XT, R=RT, Y=YT)
t1 = time.time()
print('test NLPD: %1.2f' % nlpd)

if baseline:
    with open("output/baseline_" + str(method) + "_" + str(fold) + "_nlpd.txt", "wb") as fp:
        pickle.dump(nlpd, fp)
else:
    with open("output/" + str(method) + "_" + str(fold) + "_nlpd.txt", "wb") as fp:
        pickle.dump(nlpd, fp)
