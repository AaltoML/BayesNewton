import bayesnewton
import objax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


filename = '../data/fission_normalized_counts.csv'
Y = pd.read_csv(filename, index_col=[0])
X = pd.read_csv('../data/fission_col_data.csv', index_col=[0])
X = X[['minute']]

# extract time series for one gene
genes_name = ['SPAC11D3.01c']
num = 18
x, y = X.iloc[0:num, :].values, Y.iloc[:, 0:num].loc[genes_name].values.T

# Test points
# x_test = x
x_plot = np.linspace(np.min(x)-5, np.max(x)+5, 200)
# M = 15
# z = np.linspace(np.min(x), np.max(x), M)

var_f = 15.0  # GP variance
len_f = 150.0  # GP lengthscale

kern = bayesnewton.kernels.Matern72(variance=var_f, lengthscale=len_f)
lik = bayesnewton.likelihoods.NegativeBinomial(alpha=1.0, scale=1.0)

# model = bayesnewton.models.VariationalGP(kernel=kern, likelihood=lik, X=x, Y=y)
model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.MarkovVariationalGaussNewtonGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.MarkovVariationalQuasiNewtonGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.MarkovVariationalRiemannGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=x, Y=y, power=1.)
# model = bayesnewton.models.MarkovExpectationPropagationQuasiNewtonGP(kernel=kern, likelihood=lik, X=x, Y=y, power=1.)
# model = bayesnewton.models.MarkovExpectationPropagationRiemannGP(kernel=kern, likelihood=lik, X=x, Y=y, power=1.)

lr_adam = 0.1
lr_newton = 0.25
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

print(model.likelihood.alpha)
print(model.kernel.variance)
print(model.kernel.lengthscale)

# calculate posterior predictive distribution via filtering and smoothing at train & test locations:
print('calculating the posterior predictive distribution ...')
t0 = time.time()
# posterior_mean, posterior_var = model.predict(X=x_plot)
posterior_mean_y, posterior_var_y = model.predict_y(X=x_plot)
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))

lb_y = posterior_mean_y - np.sqrt(posterior_var_y)
ub_y = posterior_mean_y + np.sqrt(posterior_var_y)

print('plotting ...')
plt.figure(1, figsize=(10, 6))
plt.clf()
plt.plot(x, y, 'b.', label='observations', clip_on=False)
plt.plot(x_plot, posterior_mean_y, 'b', label='posterior mean')
plt.fill_between(x_plot, lb_y, ub_y, color='b', alpha=0.05, label='posterior std')
plt.xlim(x_plot[0], x_plot[-1])
plt.ylim(0.0)
plt.legend()
plt.title('')
plt.xlabel('time')
plt.ylabel('gene expression')
plt.show()
