import newt
import objax
import numpy as np
import time

np.random.seed(3)


def create_grid(x1, x2, y1, y2, n1=10, n2=10):
    y = np.linspace(y1, y2, n2)
    x = np.linspace(x1, x2, n1)

    grid = []
    for i in x:
        for j in y:
            grid.append([i, j])

    return np.array(grid)


Nt_train = 5
Ns = 5
X = create_grid(0, 1, 0, 1, Nt_train, Ns)
t = np.linspace(0, 1, Nt_train, dtype=float)
R = np.tile(np.linspace(0, 1, Ns, dtype=float)[None, ...], [Nt_train, 1])

N = X.shape[0]
y = np.sin(10*X[:, 0]) + np.sin(10*X[:, 1]) + 0.01*np.random.randn(N)

# Y = y[:, None]
Y = y.reshape(Nt_train, Ns)

# print(R.shape)
# print(Y.shape)
# print(R[0].shape)
# print(X)
# print(R)
# print(R[0])

kernel_ls = [0.1, 0.2]
kernel_var = [2.2, 0.4]
likelihood_noise = 0.1

lik = newt.likelihoods.Gaussian(variance=likelihood_noise)
kern_time = newt.kernels.Matern32(variance=kernel_var[0], lengthscale=kernel_ls[0])
kern_space = newt.kernels.Matern32(variance=kernel_var[1], lengthscale=kernel_ls[1])
kern = newt.kernels.SpatioTemporalKernel(temporal_kernel=kern_time,
                                         spatial_kernel=kern_space,
                                         z=R[0],
                                         sparse=True,
                                         opt_z=False,
                                         conditional='Full')
inf = newt.inference.VariationalInference()

markov = True

if markov:
    model = newt.models.MarkovGP(kernel=kern, likelihood=lik, X=t, R=R, Y=Y)
    # model = newt.models.MarkovGP(kernel=kern, likelihood=lik, X=X, Y=y)
else:
    model = newt.models.GP(kernel=kern, likelihood=lik, X=X, Y=y)

compute_energy_and_update = objax.GradValues(inf, model.vars())

lr_adam = 0.
lr_newton = 1.
epochs = 2
opt = objax.optimizer.Adam(model.vars())


def train_op():
    model.update_posterior()
    grads, loss_ = compute_energy_and_update(model, lr=lr_newton)
    # print(grads)
    for g, var_name in zip(grads, model.vars().keys()):  # TODO: this gives wrong label to likelihood variance
        print(g, ' w.r.t. ', var_name)
    # print(model.kernel.temporal_kernel.variance)
    opt(lr_adam, grads)
    return loss_[0]


# train_op = objax.Jit(train_op, model.vars())

t0 = time.time()
for i in range(1, epochs+1):
    loss = train_op()
    print('epoch %2d: loss: %1.4f' % (i, loss))
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))
