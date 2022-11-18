import bayesnewton
import objax
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import sys

if len(sys.argv) > 1:
    inf_method = int(sys.argv[1])
    approx_method = int(sys.argv[2])
    fold = int(sys.argv[3])
    plot_final = False
else:
    inf_method = 1
    approx_method = 1
    fold = 0
    plot_final = False

model_type = 0

if model_type == 0:
    N = 400
    num_latents = 2
    num_outputs = 3
else:
    N = 150
    num_latents = 2
    num_outputs = 5
x = np.linspace(-17, 147, num=N)
y_dummy = np.zeros([N, num_outputs])

if model_type == 0:
    # noise_scale = np.array([[0.2, 0.1], [0.1, 0.3]])
    # noise_scale = np.array([[0.1, -0.075], [-0.075, 0.2]])
    noise_scale = np.array([[0.1, -0.075, -0.025], [-0.075, 0.2, 0.05], [-0.025, 0.05, 0.3]]) * 0.2
    noise_scale = noise_scale[:num_outputs, :num_outputs]
    var_f = 1.0  # GP variance
    len_f = 10.0  # GP lengthscale
    len_f_w = 70.0
else:
    np.random.seed(123)
    # construct the observation noise covariance
    temp = np.random.rand(num_outputs) - 0.4
    noise_scale = 2 * temp[:, None] @ temp[None] + 0.05 * np.eye(num_outputs)
    var_f = 1.0  # GP variance
    len_f = 5.0  # GP lengthscale
    len_f_w = 80.0

kern_latent = bayesnewton.kernels.Matern52(variance=var_f, lengthscale=len_f, fix_variance=True, fix_lengthscale=True)
kern_weight = bayesnewton.kernels.Matern52(variance=var_f, lengthscale=len_f_w, fix_variance=True, fix_lengthscale=True)
kern_latents = [kern_latent for i in range(num_latents)]
kern_weights = [kern_weight for j in range(num_latents * num_outputs)]
kern = bayesnewton.kernels.Independent(kernels=kern_latents + kern_weights)
lik = bayesnewton.likelihoods.RegressionNetwork(num_latents=num_latents,
                                                num_outputs=num_outputs,
                                                covariance=noise_scale,
                                                fix_covariance=True)

dummy_model = bayesnewton.models.MarkovGaussianProcess(kernel=kern, likelihood=lik, X=x, Y=y_dummy)
seeds = [999, 123, 321, 987]
samp = dummy_model.prior_sample(num_samps=1, X=x, seed=seeds[fold])
f_latents = samp[0, :, :num_latents]
f_weights = samp[0, :, num_latents:, 0].reshape(N, num_outputs, num_latents)
f_out = f_weights @ f_latents

np.random.seed(seeds[fold])
noise = np.linalg.cholesky(noise_scale)[None] @ np.random.multivariate_normal(np.zeros(num_outputs),
                                                                              np.eye(num_outputs),
                                                                              f_out.shape[0])[..., None]
y = np.array(f_out + noise)

x_plot = np.linspace(np.min(x)-20.0, np.max(x)+20.0, 200)

np.random.seed(123)
X, XT = x, x
Y, YT = y.flatten(), y.flatten()
Y = Y.reshape(N, -1)
YT = YT.reshape(N, -1)

if model_type == 0:
    Y[N//3:2*N//3, 1:] = np.nan
    YT[:N//3, 1:] = np.nan
    YT[2*N//3:, 1:] = np.nan
    YT[:, 0] = np.nan
    f_ground_truth = f_out[N // 3:2 * N // 3, 1:]
else:
    observed = np.array([0, 2], dtype=int)
    missing = np.array([1, 3, 4], dtype=int)
    Y[50:100, missing] = np.nan
    YT[:50, missing] = np.nan
    YT[100:, missing] = np.nan
    YT[:, observed] = np.nan

if inf_method == 0:  # Newton
    if approx_method == 0:
        model = bayesnewton.models.MarkovNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
    elif approx_method == 1:
        model = bayesnewton.models.MarkovGaussNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
    elif approx_method == 2:
        model = bayesnewton.models.MarkovQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
    elif approx_method == 3:
        model = bayesnewton.models.MarkovNewtonRiemannGP(kernel=kern, likelihood=lik, X=X, Y=Y)
if inf_method == 1:  # VI
    if approx_method == 0:
        model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=X, Y=Y)
    elif approx_method == 1:
        model = bayesnewton.models.MarkovVariationalGaussNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
    elif approx_method == 2:
        model = bayesnewton.models.MarkovVariationalQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
    elif approx_method == 3:
        model = bayesnewton.models.MarkovVariationalRiemannGP(kernel=kern, likelihood=lik, X=X, Y=Y)
elif inf_method == 2:  # PEP
    if approx_method == 0:
        model = bayesnewton.models.MarkovExpectationPropagationGP(kernel=kern, likelihood=lik, X=X, Y=Y, power=0.5)
    elif approx_method == 1:
        raise NotImplementedError
    elif approx_method == 2:
        model = bayesnewton.models.MarkovExpectationPropagationQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y, power=0.5)
    elif approx_method == 3:
        model = bayesnewton.models.MarkovExpectationPropagationRiemannGP(kernel=kern, likelihood=lik, X=X, Y=Y, power=0.5)
if inf_method == 3:  # PL
    if approx_method == 0:
        model = bayesnewton.models.MarkovPosteriorLinearisation2ndOrderGP(kernel=kern, likelihood=lik, X=X, Y=Y)
    elif approx_method == 1:
        model = bayesnewton.models.MarkovPosteriorLinearisation2ndOrderGaussNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
    elif approx_method == 2:
        model = bayesnewton.models.MarkovPosteriorLinearisation2ndOrderQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
    elif approx_method == 3:
        model = bayesnewton.models.MarkovPosteriorLinearisation2ndOrderRiemannGP(kernel=kern, likelihood=lik, X=X, Y=Y)
    elif approx_method == 4:
        model = bayesnewton.models.MarkovPosteriorLinearisationGP(kernel=kern, likelihood=lik, X=X, Y=Y)
    # elif approx_method == 5:
    #     model = bayesnewton.models.MarkovPosteriorLinearisationQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
if inf_method == 4:  # first-order VI
    model = bayesnewton.models.FirstOrderMarkovVariationalGP(kernel=kern, likelihood=lik, X=X, Y=Y)

lr_adam = 0.1
lr_newton = 0.3
iters = 1000
opt_hypers = objax.optimizer.Adam(model.vars(), beta2=0.99)  # reduce beta to make things more stable
energy = objax.GradValues(model.energy, model.vars())

unscented_transform = bayesnewton.cubature.Unscented(dim=kern.num_kernels)  # 5th-order unscented transform

damping = 0.3


@objax.Function.with_vars(model.vars() + opt_hypers.vars())
def train_op():
    if inf_method < 4:
        model.inference(lr=lr_newton, damping=damping, cubature=unscented_transform)
    dE, E = energy(cubature=unscented_transform)  # compute energy and its gradients w.r.t. hypers
    if inf_method == 4:
        opt_hypers(lr_adam, dE)
    test_nlpd_ = model.negative_log_predictive_density(X=XT, Y=YT, cubature=unscented_transform)
    posterior_mean_, posterior_cov_ = model.predict_y(X=XT[N // 3:2 * N // 3], cubature=unscented_transform)
    return E, test_nlpd_, posterior_mean_


train_op = objax.Jit(train_op)

losses = np.zeros(iters//10) * np.nan
nlpds = np.zeros(iters//10) * np.nan
rmses = np.zeros(iters//10) * np.nan

t0 = time.time()
for i in range(1, iters + 1):
    loss, test_nlpd, post_mean = train_op()
    rmse = np.sqrt(np.mean((np.squeeze(f_ground_truth) - np.squeeze(post_mean[:, 1:])) ** 2))
    print('iter %2d, energy: %1.4f, nlpd: %1.4f, rmse: %1.4f' % (i, loss[0], test_nlpd, rmse))
    if np.mod(i-1, 10) == 0:
        losses[(i-1)//10] = loss[0]
        nlpds[(i-1)//10] = test_nlpd
        rmses[(i-1)//10] = rmse
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

with open("output/gprn_" + str(inf_method) + "_" + str(approx_method) + "_" + str(fold) + "_loss.txt", "wb") as fp:
    pickle.dump(losses, fp)
with open("output/gprn_" + str(inf_method) + "_" + str(approx_method) + "_" + str(fold) + "_nlpd.txt", "wb") as fp:
    pickle.dump(nlpds, fp)
with open("output/gprn_" + str(inf_method) + "_" + str(approx_method) + "_" + str(fold) + "_rmse.txt", "wb") as fp:
    pickle.dump(rmses, fp)


if plot_final:
    t0 = time.time()
    posterior_mean, posterior_cov = model.predict_y(X=x_plot, cubature=unscented_transform)
    posterior_mean_f, posterior_cov_f = model.predict(X=x_plot)
    nlpd = model.negative_log_predictive_density(X=XT, Y=YT, cubature=unscented_transform)
    t1 = time.time()
    print('prediction time: %2.2f secs' % (t1-t0))
    print('nlpd: %2.3f' % nlpd)
    if num_outputs == 1:
        posterior_var = posterior_cov
    else:
        posterior_var = bayesnewton.utils.diag(posterior_cov)
    crosscov = posterior_cov_f[:, 0, 1]
    print('avg. cross cov', np.mean(np.abs(crosscov)))
    posterior_mean = posterior_mean.reshape(x_plot.shape[0], -1)
    posterior_var = posterior_var.reshape(x_plot.shape[0], -1)
    lb = posterior_mean - 1.96 * posterior_var ** 0.5
    ub = posterior_mean + 1.96 * posterior_var ** 0.5

    # f0rmse = np.sqrt(np.mean((f_out[:, 0, 0]-posterior_mean[:, 0])**2))
    # print('RMSE 1: ', f0rmse)
    # if num_outputs > 1:
    #     f1rmse = np.sqrt(np.mean((f_out[:, 1, 0]-posterior_mean[:, 1])**2))
    #     print('RMSE 2: ', f1rmse)
    # if num_outputs > 2:
    #     f2rmse = np.sqrt(np.mean((f_out[:, 2, 0]-posterior_mean[:, 2])**2))
    #     print('RMSE 3: ', f2rmse)

    # _, _, hessian = vmap(model.likelihood.log_likelihood_gradients)(  # parallel
    #     model.Y,
    #     model.posterior_mean.value
    # )
    #
    # outliers = np.argwhere(np.squeeze(hessian > 0))

    print('plotting ...')
    plt.figure(1, figsize=(12, 5))
    plt.clf()
    plt.plot(X, Y[:, 0], 'b.', alpha=0.3)
    plt.plot(X, YT[:, 0], 'b*', markersize=4)
    plt.plot(x_plot, posterior_mean[:, 0], 'b')
    # plt.plot(x_plot, f1_plot, 'b--')
    plt.fill_between(x_plot, lb[:, 0], ub[:, 0], color='b', alpha=0.05)
    if num_outputs > 1:
        plt.plot(X, Y[:, 1], 'r.', alpha=0.3)
        plt.plot(X, YT[:, 1], 'r*', markersize=4)
        plt.plot(x_plot, posterior_mean[:, 1], 'r')
        # plt.plot(x_plot, f2_plot, 'r--')
        plt.fill_between(x_plot, lb[:, 1], ub[:, 1], color='r', alpha=0.05)
    if num_outputs > 2:
        plt.plot(X, Y[:, 2], 'g.', alpha=0.3)
        plt.plot(X, YT[:, 2], 'g*', markersize=4)
        plt.plot(x_plot, posterior_mean[:, 2], 'g')
        # plt.plot(x_plot, f3_plot, 'g--')
        plt.fill_between(x_plot, lb[:, 2], ub[:, 2], color='g', alpha=0.05)
    if num_outputs > 3:
        plt.plot(X, Y[:, 3], 'k.', alpha=0.3)
        plt.plot(X, YT[:, 3], 'k*', markersize=4)
        plt.plot(x_plot, posterior_mean[:, 3], 'k')
        # plt.plot(x_plot, f4_plot, 'g--')
        plt.fill_between(x_plot, lb[:, 3], ub[:, 3], color='k', alpha=0.05)
    if num_outputs > 4:
        plt.plot(X, Y[:, 4], 'c.', alpha=0.3)
        plt.plot(X, YT[:, 4], 'c*', markersize=4)
        plt.plot(x_plot, posterior_mean[:, 4], 'c')
        # plt.plot(x_plot, f5_plot, 'g--')
        plt.fill_between(x_plot, lb[:, 4], ub[:, 4], color='c', alpha=0.05)
    # plt.plot(x_test, y_test, 'r.', alpha=0.4, label='test observations')
    # plt.plot(x[outliers], y[outliers], 'g*', label='outliers')
    plt.xlim([x_plot[0], x_plot[-1]])
    if hasattr(model, 'Z'):
        plt.plot(model.Z.value[:, 0], -2 * np.ones_like(model.Z.value[:, 0]), 'b^', markersize=5)
    plt.legend()
    plt.title('GP Regression Newtork')
    plt.xlabel('$X$')
    plt.show()
