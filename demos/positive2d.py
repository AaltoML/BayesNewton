import bayesnewton
import objax
import numpy as np
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from convertbng.util import convert_bng
import time


def datetime_to_epoch(datetime):
    """
        Converts a datetime to a number
        args:
            datatime: is a pandas column
    """
    return datetime.astype('int64') // 1e9


species = 'pm10'

raw_data = pd.read_csv('../data/aq_data.csv')
sites_df = pd.read_csv('../data/laqn_sites.csv', sep=';')

# filter sites not in london
london_box = [
    [51.279, 51.684],  # lat
    [-0.533, 0.208]  # lon
]

sites_df = sites_df[(sites_df['Latitude'] > london_box[0][0]) & (sites_df['Latitude'] < london_box[0][1])]
sites_df = sites_df[(sites_df['Longitude'] > london_box[1][0]) & (sites_df['Longitude'] < london_box[1][1])]

# merge spatial infomation to data
raw_data = raw_data.merge(sites_df, left_on='site', right_on='SiteCode')

# convert to datetimes
raw_data['date'] = pd.to_datetime(raw_data['date'])
raw_data['epoch'] = datetime_to_epoch(raw_data['date'])

# get data in date range
data_range_start = '2019/02/01 00:00:00'
data_range_end = '2019/02/01 04:00:00'  # '2019/02/01 23:59:59'  # '2019/02/25 23:59:59', '2019/03/11 23:59:59', '2019/04/17 23:59:59'

raw_data = raw_data[(raw_data['date'] >= data_range_start) & (raw_data['date'] < data_range_end)]

Xraw = np.array(raw_data[['epoch', 'Longitude', 'Latitude']])
Yraw = np.array(raw_data[[species]])

Xraw = Xraw[~np.isnan(np.squeeze(Yraw))]
Yraw = Yraw[~np.isnan(np.squeeze(Yraw))]

X_scaler = StandardScaler().fit(Xraw)
Xraw = X_scaler.transform(Xraw)

scale_y = 30.
Yraw = Yraw / scale_y
# Y_scaler = StandardScaler().fit(Yraw)
# Yraw = Y_scaler.transform(Yraw)

# plt.plot(Yraw)
# plt.show()

print('N =', Yraw.shape[0])

fold = 0

np.random.seed(123)
# 4-fold cross-validation setup
ind_shuffled = np.random.permutation((Yraw.shape[0] // 4) * 4)
ind_split = np.stack(np.split(ind_shuffled, 4))  # 4 random batches of data indices

# Get training and test indices
ind_test = ind_split[fold]  # np.sort(ind_shuffled[:N//4])
ind_train = np.concatenate(ind_split[np.arange(4) != fold])
X = Xraw[ind_train]  # 75/25 train/test split
XT = Xraw[ind_test]
Y = Yraw[ind_train]
YT = Yraw[ind_test]

# X = X[:500]
# XT = XT[:150]
# Y = Y[:500]
# YT = YT[:150]

M = 100

Z = kmeans2(X, M, minit="points")[0]

kern_process_ = bayesnewton.kernels.Matern52(variance=1.0, lengthscale=1.0)
kern_process = bayesnewton.kernels.Separable([kern_process_, kern_process_, kern_process_])
kern_noise_ = bayesnewton.kernels.Matern52(variance=1.0, lengthscale=1.0)
kern_noise = bayesnewton.kernels.Separable([kern_noise_, kern_noise_, kern_noise_])
kern = bayesnewton.kernels.Independent([kern_process, kern_noise])
# lik = bayesnewton.likelihoods.Positive(variance=0.25)
# lik = bayesnewton.likelihoods.PositiveStudentsT(scale=0.25)
# lik = bayesnewton.likelihoods.HeteroscedasticNoise()
lik = bayesnewton.likelihoods.HeteroscedasticStudentsT(df=5.)
# model = bayesnewton.models.VariationalGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model = bayesnewton.models.ExpectationPropagationGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model = bayesnewton.models.VariationalGaussNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
model = bayesnewton.models.VariationalQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model = bayesnewton.models.ExpectationPropagationQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y)
# model = bayesnewton.models.SparseVariationalGP(kernel=kern, likelihood=lik, X=X, Y=Y, Z=Z, opt_z=True)
# model = bayesnewton.models.SparseVariationalGaussNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y, Z=Z, opt_z=True)
# model = bayesnewton.models.SparseVariationalQuasiNewtonGP(kernel=kern, likelihood=lik, X=X, Y=Y, Z=Z, opt_z=True)

lr_adam = 0.01
lr_newton = 0.1  # 0.05
iters = 1000
iters_warmup = 500
opt_hypers = objax.optimizer.Adam(model.vars())
energy = objax.GradValues(model.energy, model.vars())

# damping = np.logspace(np.log10(0.8), np.log10(1e-1), num=iters)
# damping = np.linspace(0.8, 1e-3, num=iters)
# damping = np.linspace(0.5, 0.1, num=iters)
# damping = np.linspace(0.5, 0.1, num=iters)
# damping = np.logspace(np.log10(0.8), np.log10(0.01), num=iters)
damping = 0.1
damping_warmup = 0.5


@objax.Function.with_vars(model.vars() + opt_hypers.vars())
def train_op():
    model.inference(lr=lr_newton, damping=damping)  # perform inference and update variational params
    dE, E = energy()  # compute energy and its gradients w.r.t. hypers
    opt_hypers(lr_adam, dE)
    test_nlpd_ = model.negative_log_predictive_density(X=XT, Y=YT)
    return E, test_nlpd_


@objax.Function.with_vars(model.vars() + opt_hypers.vars())
def train_op_warmup():
    model.inference(lr=lr_newton, damping=damping_warmup)  # perform inference and update variational params
    dE, E = energy()  # compute energy and its gradients w.r.t. hypers
    # opt_hypers(lr_adam, dE)
    test_nlpd_ = model.negative_log_predictive_density(X=XT, Y=YT)
    return E, test_nlpd_


train_op = objax.Jit(train_op)
train_op_warmup = objax.Jit(train_op_warmup)

t0 = time.time()
for i in range(1, iters_warmup + 1):
    loss, test_nlpd = train_op_warmup()
    print('iter %2d, energy: %1.4f, nlpd: %1.4f' % (i, loss[0], test_nlpd))
for i in range(1, iters + 1):
    loss, test_nlpd = train_op()
    print('iter %2d, energy: %1.4f, nlpd: %1.4f' % (i, loss[0], test_nlpd))
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))
