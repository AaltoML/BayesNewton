import bayesnewton
import objax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


y = pd.read_csv('../data/normalized_alpha_counts.csv', index_col=[0])
x = pd.read_csv('../data/alpha_time_points.csv', index_col=[0])

y = y.rename(index={'ENSMUSG00000015879': 'Fam184b', 'ENSMUSG00000059173': 'Pde1a'})
genes_name = ['Fam184b', 'Pde1a']

x, y = x.values, y.loc[genes_name].values.T

y = y[:, :1]  # Fam184b
# y = y[:, 1:]  # Pde1a

# Test points
# x_test = x
x_plot = np.linspace(np.min(x)-0.1, np.max(x)+0.1, 200)

var_f = 10.0  # GP variance
len_f = 1.0  # GP lengthscale

kern = bayesnewton.kernels.Matern72(variance=var_f, lengthscale=len_f)
lik = bayesnewton.likelihoods.NegativeBinomial(alpha=1.0, scale=1.0)
# lik = bayesnewton.likelihoods.ZeroInflatedNegativeBinomial(alpha=1.0, km=1.0)


# model = bayesnewton.models.VariationalGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.ExpectationPropagationGP(kernel=kern, likelihood=lik, X=x, Y=y)
model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=x, Y=y)
# model = bayesnewton.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=x, Y=y)

lr_adam = 0.1
lr_newton = 1
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
# plt.plot(x_plot, lik.link_fn(posterior_mean), 'b--', label='posterior mean')
plt.fill_between(x_plot, lb_y, ub_y, color='b', alpha=0.05, label='posterior std')
plt.xlim(x_plot[0], x_plot[-1])
plt.ylim(0.0)
plt.legend()
plt.title('')
plt.xlabel('time')
plt.ylabel('gene expression')
plt.show()
