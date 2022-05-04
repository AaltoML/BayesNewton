import bayesnewton
import objax
import numpy as np

# import pods
import matplotlib.pyplot as plt
import time


# data = pods.datasets.olympic_marathon_men()
data = np.loadtxt("../data/olympicMarathonTimes.csv", delimiter=",")
x = data[:, :1]
y = data[:, 1:]

x_train = x[:-2, :]
y_train = y[:-2, :]

x_test = x[-2:, :]
y_test = y[-2:, :]

offset = y_train.mean()
scale = np.sqrt(y_train.var())

# remove outlier
# y[2] = np.nan

xlim = (1875, 2030)
ylim = (2.5, 6.5)
yhat = (y_train - offset) / scale

np.random.seed(12345)
x_plot = np.linspace(xlim[0], xlim[1], 200)[:, None]

var_f = 1.0  # GP variance
len_f = 40  # GP lengthscale
var_y = 0.5  # observation noise

kern = bayesnewton.kernels.SquaredExponential(variance=var_f, lengthscale=len_f)
lik = bayesnewton.likelihoods.Gaussian(variance=var_y)
model = bayesnewton.models.VariationalGP(kernel=kern, likelihood=lik, X=x_train, Y=yhat)


lr_adam = 0.1
lr_newton = 1.0
iters = 100
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
    print("iter %2d, energy: %1.4f" % (i, loss[0]))
t1 = time.time()
print("optimisation time: %2.2f secs" % (t1 - t0))

t0 = time.time()
posterior_mean, posterior_var = model.predict_y(X=x_plot)
t1 = time.time()
print("prediction time: %2.2f secs" % (t1 - t0))
lb = posterior_mean - 2 * posterior_var**0.5
ub = posterior_mean + 2 * posterior_var**0.5

print("plotting ...")
plt.figure(1, figsize=(8, 4))
plt.clf()
plt.plot(x_train, y_train, "k.", label="training observations")
plt.plot(x_test, y_test, "gx", label="held out observations")
plt.plot(x_plot, posterior_mean * scale + offset, "r", label="posterior mean")
plt.fill_between(
    x_plot[:, 0],
    lb * scale + offset,
    ub * scale + offset,
    color="r",
    alpha=0.05,
    label="2 std",
)
plt.xlim([x_plot[0], x_plot[-1]])
plt.ylim([2.8, 5.5])
plt.legend(loc=1)
# plt.xticks([-2, -1, 0., 1., 2])
# plt.yticks([-0.5, 0., 0.5, 1., 1.5])
plt.title("GP regression - Olympic Marathon Data")
plt.xlabel("Year")
plt.ylabel("Pace, min / km")
# plt.savefig('/Users/wilkinw1/postdoc/gp_course/lec8_deepgps/marathon.png')
# plt.savefig('/Users/wilkinw1/postdoc/gp_course/lec8_deepgps/marathon_outlier_removed.png')
plt.show()
