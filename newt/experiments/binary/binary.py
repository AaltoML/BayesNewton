import sys
import newt
import objax
import numpy as np
import time
import pickle

print('generating some data ...')
np.random.seed(99)
N = 10000  # number of points
x = np.sort(70 * np.random.rand(N))
sn = 0.01
f = lambda x_: 12. * np.sin(4 * np.pi * x_) / (0.25 * np.pi * x_ + 1)
y_ = f(x) + np.math.sqrt(sn)*np.random.randn(x.shape[0])
y = np.sign(y_)
y[y == -1] = 0

ind_shuffled = np.random.permutation(N)
ind_split = np.stack(np.split(ind_shuffled, 10))  # 10 random batches of data indices

if len(sys.argv) > 1:
    method = int(sys.argv[1])
    fold = int(sys.argv[2])
else:
    method = 4
    fold = 0

print('method number', method)
print('batch number', fold)

# Get training and test indices
ind_test = ind_split[fold]  # np.sort(ind_shuffled[:N//10])
ind_train = np.concatenate(ind_split[np.arange(10) != fold])

x *= 100

x_train = x[ind_train]  # 90/10 train/test split
x_test = x[ind_test]
y_train = y[ind_train]
y_test = y[ind_test]
N = x_train.shape[0]  # number of points
batch_size = N  # 2000
M = 1000
z = np.linspace(x[0], x[-1], M)

if len(sys.argv) > 3:
    baseline = int(sys.argv[3])
else:
    baseline = 0

# if baseline:
#     batch_size = N

var_f = 1.  # GP variance
len_f = 25.  # GP lengthscale

kern = newt.kernels.Matern72(variance=var_f, lengthscale=len_f)
lik = newt.likelihoods.Bernoulli(link='logit')

if method == 0:
    inf = newt.inference.Taylor()
elif method == 1:
    inf = newt.inference.PosteriorLinearisation()
elif method == 2:
    inf = newt.inference.ExpectationPropagation(power=1)
elif method == 3:
    inf = newt.inference.ExpectationPropagation(power=0.5)
elif method == 4:
    inf = newt.inference.ExpectationPropagation(power=0.01)
elif method == 5:
    inf = newt.inference.VariationalInference()

if baseline:
    model = newt.models.MarkovGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train)
else:
    model = newt.models.SparseMarkovGP(kernel=kern, likelihood=lik, X=x_train, Y=y_train, Z=z)

trainable_vars = model.vars() + inf.vars()
energy = objax.GradValues(inf.energy, trainable_vars)

lr_adam = 0.1
lr_newton = 0.5
iters = 500
opt = objax.optimizer.Adam(trainable_vars)


def train_op():
    batch = np.random.permutation(N)[:batch_size]
    inf(model, lr=lr_newton, batch_ind=batch)  # perform inference and update variational params
    dE, E = energy(model, batch_ind=batch)  # compute energy and its gradients w.r.t. hypers
    return dE, E


train_op = objax.Jit(train_op, trainable_vars)


t0 = time.time()
for i in range(1, iters + 1):
    grad, loss = train_op()
    opt(lr_adam, grad)
    print('iter %2d, energy: %1.4f' % (i, loss[0]))
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

print('calculating the posterior predictive distribution ...')
t0 = time.time()
nlpd = model.negative_log_predictive_density(X=x_test, Y=y_test)
t1 = time.time()
print('nlpd: %2.3f' % nlpd)

if baseline:
    with open("output/baseline_" + str(method) + "_" + str(fold) + "_nlpd.txt", "wb") as fp:
        pickle.dump(nlpd, fp)
else:
    with open("output/" + str(method) + "_" + str(fold) + "_nlpd.txt", "wb") as fp:
        pickle.dump(nlpd, fp)
