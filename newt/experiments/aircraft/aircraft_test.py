import sys
sys.path.insert(0, '../../')
import numpy as np
from jax.experimental import optimizers
import matplotlib.pyplot as plt
import time
import pandas as pd
from sde_gp import SDEGP
import approximate_inference as approx_inf
import priors
import likelihoods
from utils import softplus_list, plot
from datetime import date
import pickle

plot_final = False
plot_intermediate = False

print('loading data ...')
aircraft_accidents = pd.read_csv('aircraft_accidents.txt', sep='-', header=None).values

num_data = aircraft_accidents.shape[0]
xx = np.zeros([num_data, 1])
for j in range(num_data):
    xx[j] = date.toordinal(date(aircraft_accidents[j, 0], aircraft_accidents[j, 1], aircraft_accidents[j, 2])) + 366

BIN_WIDTH = 1
# Discretize the data
x_min = np.floor(np.min(xx))
x_max = np.ceil(np.max(xx))
x_max_int = x_max-np.mod(x_max-x_min, BIN_WIDTH)
x = np.linspace(x_min, x_max_int, num=int((x_max_int-x_min)/BIN_WIDTH+1))
x = np.concatenate([np.min(x)-np.linspace(61, 1, num=61), x])  # pad with zeros to reduce strange edge effects
y, _ = np.histogram(xx, np.concatenate([[-1e10], x[1:]-np.diff(x)/2, [1e10]]))
N = y.shape[0]
print('N =', N)

np.random.seed(123)
ind_shuffled = np.random.permutation(N)
ind_split = np.stack(np.split(ind_shuffled, 10))  # 10 random batches of data indices

np.random.seed(123)
# meanval = np.log(len(disaster_timings)/num_time_bins)  # TODO: incorporate mean

if len(sys.argv) > 1:
    method = int(sys.argv[1])
    fold = int(sys.argv[2])
else:
    method = 0
    fold = 0

print('method number', method)
print('batch number', fold)

# Get training and test indices
ind_test = ind_split[fold]  # np.sort(ind_shuffled[:N//10])
ind_train = np.concatenate(ind_split[np.arange(10) != fold])
x_train = x  # [ind_train]  # 90/10 train/test split
x_test = x  # [ind_test]
y_train = y  # [ind_train]
y_test = y  # [ind_test]
N_batch = 5000
M = 5000
# z = np.linspace(701050, 737050, M)
z = np.linspace(x[0], x[-1], M)

prior_1 = priors.Matern52(variance=2., lengthscale=5.5e4)
prior_2 = priors.QuasiPeriodicMatern32(variance=1., lengthscale_periodic=2., period=365., lengthscale_matern=1.5e4)
prior_3 = priors.QuasiPeriodicMatern32(variance=1., lengthscale_periodic=2., period=7., lengthscale_matern=30*365.)

prior = priors.Sum([prior_1, prior_2, prior_3])
lik = likelihoods.Poisson()

if method == 0:
    inf_method = approx_inf.EKS(damping=.5)
elif method == 1:
    inf_method = approx_inf.UKS(damping=.5)
elif method == 2:
    inf_method = approx_inf.GHKS(damping=.5)
elif method == 3:
    inf_method = approx_inf.EP(power=1, intmethod='GH', damping=.5)
elif method == 4:
    inf_method = approx_inf.EP(power=0.5, intmethod='GH', damping=.5)
elif method == 5:
    inf_method = approx_inf.EP(power=0.01, intmethod='GH', damping=.5)
elif method == 6:
    inf_method = approx_inf.VI(intmethod='GH', damping=.5)

model = SDEGP(prior=prior, likelihood=lik, t=x_train, y=y_train, t_test=x_test, y_test=y_test,
              approx_inf=inf_method, z=z)

opt_init, opt_update, get_params = optimizers.adam(step_size=1e-1)
# parameters should be a 2-element list [param_prior, param_likelihood]
opt_state = opt_init([model.prior.hyp, model.likelihood.hyp])


def gradient_step(i, state, mod):
    params = get_params(state)
    mod.prior.hyp = params[0]
    mod.likelihood.hyp = params[1]

    # grad(Filter) + Smoother:
    batch_ind = np.random.permutation(N)[:N_batch]

    # grad(Filter) + Smoother:
    neg_log_marg_lik, gradients = mod.run(batch_ind=batch_ind)
    nlml = neg_log_marg_lik * N / N_batch

    print('iter %2d: nlml=%2.2f' %
          (i, nlml))

    if plot_intermediate:
        plot(mod, i)

    return opt_update(i, gradients, state)


print('optimising the hyperparameters ...')
t0 = time.time()
for j in range(2):
    opt_state = gradient_step(j, opt_state, model)
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

# calculate posterior predictive distribution via filtering and smoothing at train & test locations:
print('calculating the posterior predictive distribution ...')
t0 = time.time()
posterior_mean, posterior_cov, inducing_mean, inducing_cov, nlpd = model.predict()
mu = np.squeeze(posterior_mean)
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))
print('NLPD: %1.2f' % nlpd)

# with open("output/" + str(method) + "_" + str(fold) + "_nlpd.txt", "wb") as fp:
#     pickle.dump(nlpd, fp)

# with open("output/" + str(method) + "_" + str(fold) + "_nlpd.txt", "rb") as fp:
#     nlpd_show = pickle.load(fp)
# print(nlpd_show)

test_id = model.test_id
t_test = model.t_all[test_id]
link_fn = model.likelihood.link_fn
scale = N / (max(t_test) - min(t_test))
post_mean_lgcp = link_fn(posterior_mean[:, 0, 0] + posterior_cov[:, 0, 0] / 2) * scale
lb_lgcp = link_fn(posterior_mean[:, 0, 0] - np.sqrt(posterior_cov[:, 0, 0]) * 1.645) * scale
ub_lgcp = link_fn(posterior_mean[:, 0, 0] + np.sqrt(posterior_cov[:, 0, 0]) * 1.645) * scale
inducing_mean_lgcp = link_fn(inducing_mean[:, 0, 0] + inducing_cov[:, 0, 0] / 2) * scale

plt.figure(1)
# plt.plot(t_test, mu, 'b-')
plt.plot(t_test, post_mean_lgcp, 'g', label='posterior mean')
plt.fill_between(t_test[..., 0], lb_lgcp, ub_lgcp, color='g', alpha=0.05, label='95% confidence')
plt.plot(z, inducing_mean_lgcp, 'g.', label='inducing mean', markersize=8)
plt.show()
