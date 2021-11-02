import torch
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import MaternKernel, ScaleKernel, GridInterpolationKernel
from gpytorch.distributions import MultivariateNormal
import numpy as np
from loguru import logger
import pickle
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import sys


if len(sys.argv) > 1:
    ind = int(sys.argv[1])
    plot_final = False
else:
    ind = 0
    plot_final = True


inducing_type = 'all_time'  # 'default'
num_z = 30
likelihood_noise = 5.
kernel_lengthscales = [0.001, 0.2, 0.2]
step_size = 0.01
iters = 300
init_params = {}
optimizer = torch.optim.Adam

cpugpu = str(0)


# ===========================Load Data===========================
train_data = pickle.load(open("data/train_data_" + str(ind) + ".pickle", "rb"))
pred_data = pickle.load(open("data/pred_data_" + str(ind) + ".pickle", "rb"))

X = train_data['X']
Y = np.squeeze(train_data['Y'])

X_t = pred_data['test']['X']
Y_t = pred_data['test']['Y']

print('X: ', X.shape)

non_nan_idx = np.squeeze(~np.isnan(Y))

X = torch.tensor(X[non_nan_idx]).float()
Y = torch.tensor(Y[non_nan_idx]).float()

D = X.shape[1]
Nt = 2159  # number of time steps

non_nan_idx_t = np.squeeze(~np.isnan(Y_t))

X_t = torch.tensor(X_t[non_nan_idx_t]).float()
Y_t = np.squeeze(Y_t[non_nan_idx_t])


class GPRegressionModelSKI(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, kernel, likelihood):
        super(GPRegressionModelSKI, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()

        self.base_covar_module = kernel
        self.base_covar_module.lengthscale = torch.tensor(kernel_lengthscales)
        logger.info(f'kernel_lengthscales : {kernel_lengthscales}')

        if inducing_type == 'default':
            grid_size = gpytorch.utils.grid.choose_grid_size(train_x)
            init_params['grid_size'] = grid_size

        elif inducing_type == 'all_time':
            grid_size = np.array([Nt, np.ceil(np.sqrt(num_z)), np.ceil(np.sqrt(num_z))]).astype(int)
            init_params['grid_size'] = grid_size

        logger.info(f'grid_size : {grid_size}')

        self.covar_module = ScaleKernel(
            GridInterpolationKernel(self.base_covar_module, grid_size, num_dims=D)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


kern = MaternKernel(ard_num_dims=3, nu=1.5)
# kern = MaternKernel(ard_num_dims=1, nu=1.5)
lik = gpytorch.likelihoods.GaussianLikelihood()
lik.noise = torch.tensor(likelihood_noise)

model = GPRegressionModelSKI(X, Y, kern, lik)  # SKI model

# train
model.train()
lik.train()

# Use the adam optimizer
optimizer = optimizer(model.parameters(), lr=step_size)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)

loss_arr = []


def train():

    for i in range(iters):
        # Zero backprop gradients
        optimizer.zero_grad()

        with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_root_decomposition_size(30):

            # Get output from model
            output = model(X)

            # Calc loss and backprop derivatives
            loss = -mll(output, torch.squeeze(Y))
            loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, iters, loss.item()))

        loss_arr.append(loss.detach().numpy())

        optimizer.step()
        torch.cuda.empty_cache()


start = timer()

with gpytorch.settings.use_toeplitz(True):
    train()

end = timer()

training_time = end - start


# ===========================Predict===========================

model.eval()
lik.eval()

print('noise var:', model.likelihood.noise.detach().numpy())

logger.info('Predicting')


with gpytorch.settings.max_preconditioner_size(10), torch.no_grad():
    with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_root_decomposition_size(30), gpytorch.settings.fast_pred_var():
        preds = model(X_t)


def negative_log_predictive_density(y, post_mean, post_cov, lik_cov):
    # logZₙ = log ∫ 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ = log 𝓝(yₙ|mₙ,σ²+vₙ)
    cov = lik_cov + post_cov
    lZ = np.squeeze(-0.5 * np.log(2 * np.pi * cov) - 0.5 * (y - post_mean) ** 2 / cov)
    return -lZ


posterior_mean, posterior_var = preds.mean.detach().numpy(), preds.variance.detach().numpy()

noise_var = model.likelihood.noise.detach().numpy()
print('noise var:', noise_var)

nlpd = np.mean(negative_log_predictive_density(y=Y_t,
                                               post_mean=posterior_mean,
                                               post_cov=posterior_var,
                                               lik_cov=noise_var))
rmse = np.sqrt(np.nanmean((np.squeeze(Y_t) - np.squeeze(posterior_mean))**2))
print('nlpd: %2.3f' % nlpd)
print('rmse: %2.3f' % rmse)

avg_time_taken = training_time / iters
print('avg iter time:', avg_time_taken)

with open("output/ski_" + str(ind) + "_" + cpugpu + "_time.txt", "wb") as fp:
    pickle.dump(avg_time_taken, fp)
with open("output/ski_" + str(ind) + "_" + cpugpu + "_nlpd.txt", "wb") as fp:
    pickle.dump(nlpd, fp)
with open("output/ski_" + str(ind) + "_" + cpugpu + "_rmse.txt", "wb") as fp:
    pickle.dump(rmse, fp)

if plot_final:
    plt.plot(posterior_mean)
    plt.show()
