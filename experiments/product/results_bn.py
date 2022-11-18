import pickle
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

make_tikz = True

num_methods = 4
num_approx = 6
num_folds = 4
num_iters = 30

method_loss = np.zeros([num_methods, num_approx, num_folds, num_iters]) * np.nan
for method in range(num_methods):
    for approx in range(num_approx):
        for fold in range(num_folds):
            if approx == 5:
                if method == 1:
                    with open("output/product_4_0_" + str(fold) + "_loss.txt", "rb") as fp:
                        result = pickle.load(fp)
                        for i in range(1, num_iters):
                            if np.isnan(result[i]):
                                result[i] = result[i - 1]
                        method_loss[method, approx, fold] = result[:num_iters]
            elif not (method == 2 and approx == 1):
                if method > 2 or approx < 4:
                    with open("output/product_" + str(method) + "_" + str(approx) + "_" + str(fold) + "_loss.txt", "rb") as fp:
                        result = pickle.load(fp)
                        for i in range(1, num_iters):
                            if np.isnan(result[i]):
                                result[i] = result[i-1]
                        method_loss[method, approx, fold] = result[:num_iters]

np.set_printoptions(precision=3)
loss_mean = np.mean(method_loss, axis=2)
# print(loss_mean)
# print(np.nanmean(method_loss, axis=2))
np.set_printoptions(precision=2)
loss_std = np.std(method_loss, axis=2)
# print(loss_std)
# print(np.nanstd(method_loss, axis=2))

method_nlpd = np.zeros([num_methods, num_approx, num_folds, num_iters]) * np.nan
for method in range(num_methods):
    for approx in range(num_approx):
        for fold in range(num_folds):
            if approx == 5:
                if method == 1:
                    with open("output/product_4_0_" + str(fold) + "_nlpd.txt", "rb") as fp:
                        result = pickle.load(fp)
                        for i in range(1, num_iters):
                            if np.isnan(result[i]):
                                result[i] = result[i - 1]
                        method_nlpd[method, approx, fold] = result[:num_iters]
            elif not (method == 2 and approx == 1):
                if method > 2 or approx < 4:
                    with open("output/product_" + str(method) + "_" + str(approx) + "_" + str(fold) + "_nlpd.txt", "rb") as fp:
                        result = pickle.load(fp)
                        for i in range(1, num_iters):
                            if np.isnan(result[i]):
                                result[i] = result[i-1]
                        method_nlpd[method, approx, fold] = result[:num_iters]

np.set_printoptions(precision=3)
nlpd_mean = np.mean(method_nlpd, axis=2)
# print(nlpd_mean)
# print(np.nanmean(method_nlpd, axis=2))
np.set_printoptions(precision=2)
nlpd_std = np.std(method_nlpd, axis=2)
# print(nlpd_std)
# print(np.nanstd(method_nlpd, axis=2))

method_rmse = np.zeros([num_methods, num_approx, num_folds, num_iters]) * np.nan
for method in range(num_methods):
    for approx in range(num_approx):
        for fold in range(num_folds):
            if approx == 5:
                if method == 1:
                    with open("output/product_4_0_" + str(fold) + "_rmse.txt", "rb") as fp:
                        result = pickle.load(fp)
                        for i in range(1, num_iters):
                            if np.isnan(result[i]):
                                result[i] = result[i - 1]
                        method_rmse[method, approx, fold] = result[:num_iters]
            elif not (method == 2 and approx == 1):
                if method > 2 or approx < 4:
                    with open("output/product_" + str(method) + "_" + str(approx) + "_" + str(fold) + "_rmse.txt", "rb") as fp:
                        result = pickle.load(fp)
                        for i in range(1, num_iters):
                            if np.isnan(result[i]):
                                result[i] = result[i-1]
                        method_rmse[method, approx, fold] = result[:num_iters]

np.set_printoptions(precision=3)
rmse_mean = np.mean(method_rmse, axis=2)
# print(nlpd_mean)
# print(np.nanmean(method_rmse, axis=2))
np.set_printoptions(precision=2)
rmse_std = np.std(method_rmse, axis=2)
# print(nlpd_std)
# print(np.nanstd(method_rmse, axis=2))

print(loss_mean.shape)
iter_num = np.arange(num_iters) * 10

plot_iters = 50

# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })
fig_width = 8
fig_height = 5

linewidth = 3
linestyleh = '--'
linestyleg = '-'
linestyleq = '-'
linestyler = '--'
linestylef = '-.'

losslims = [-50, 1000]
nlpdlims = [-0.2, 0.6]
rmselims = [0., 1.4]

method = 0

# plt.figure(1, figsize=(fig_width, fig_height))
plt.figure(1)
plt.plot(iter_num, loss_mean[method, 0].T, linewidth=linewidth, linestyle=linestyleh, label='heuristic Newton')
plt.plot(iter_num, loss_mean[method, 3].T, linewidth=linewidth, linestyle=linestyler, label='Riemannian grads Newton')
plt.plot(iter_num, loss_mean[method, 1].T, linewidth=linewidth, linestyle=linestyleg, label='Gauss-Newton')
plt.plot(iter_num, loss_mean[method, 2].T, linewidth=linewidth, linestyle=linestyleq, label='quasi-Newton')
plt.title('Training Loss (Newton)')
plt.xlabel('iteration number')
plt.ylim(losslims)
plt.gca().tick_params(axis='both', direction='in')
plt.legend()
if make_tikz:
    tikzplotlib.save('/Users/wilkinw1/postdoc/inprogress/newton-smoothers/paper/fig/product-newton-loss.tex',
                     axis_width='\\figurewidth',
                     axis_height='\\figureheight',
                     tex_relative_path_to_data='./fig/')

# plt.figure(2, figsize=(fig_width, fig_height))
plt.figure(2)
plt.plot(iter_num, nlpd_mean[method, 0].T, linewidth=linewidth, linestyle=linestyleh, label='heuristic Newton')
plt.plot(iter_num, nlpd_mean[method, 3].T, linewidth=linewidth, linestyle=linestyler, label='Riemannian grads Newton')
plt.plot(iter_num, nlpd_mean[method, 1].T, linewidth=linewidth, linestyle=linestyleg, label='Gauss-Newton')
plt.plot(iter_num, nlpd_mean[method, 2].T, linewidth=linewidth, linestyle=linestyleq, label='quasi-Newton')
plt.title('Test NLPD (Newton)')
plt.xlabel('iteration number')
plt.gca().tick_params(axis='both', direction='in')
plt.ylim(nlpdlims)
# plt.legend()
if make_tikz:
    tikzplotlib.save('/Users/wilkinw1/postdoc/inprogress/newton-smoothers/paper/fig/product-newton-nlpd.tex',
                     axis_width='\\figurewidth',
                     axis_height='\\figureheight',
                     tex_relative_path_to_data='./fig/')

plt.figure(3)
plt.plot(iter_num, rmse_mean[method, 0].T, linewidth=linewidth, linestyle=linestyleh, label='heuristic Newton')
plt.plot(iter_num, rmse_mean[method, 3].T, linewidth=linewidth, linestyle=linestyler, label='Riemannian grads Newton')
plt.plot(iter_num, rmse_mean[method, 1].T, linewidth=linewidth, linestyle=linestyleg, label='Gauss-Newton')
plt.plot(iter_num, rmse_mean[method, 2].T, linewidth=linewidth, linestyle=linestyleq, label='quasi-Newton')
plt.title('Ground Truth RMSE (Newton)')
plt.xlabel('iteration number')
plt.gca().tick_params(axis='both', direction='in')
plt.ylim(rmselims)
# plt.legend()
if make_tikz:
    tikzplotlib.save('/Users/wilkinw1/postdoc/inprogress/newton-smoothers/paper/fig/product-newton-rmse.tex',
                     axis_width='\\figurewidth',
                     axis_height='\\figureheight',
                     tex_relative_path_to_data='./fig/')

method = 1

# plt.figure(3, figsize=(fig_width, fig_height))
plt.figure(4)
plt.plot(iter_num, loss_mean[method, 0].T, linewidth=linewidth, linestyle=linestyleh, label='heuristic VI')
plt.plot(iter_num, loss_mean[method, 3].T, linewidth=linewidth, linestyle=linestyler, label='Riemannian grads VI')
plt.plot(iter_num, loss_mean[method, 1].T, linewidth=linewidth, linestyle=linestyleg, label='variational Gauss-Newton')
plt.plot(iter_num, loss_mean[method, 2].T, linewidth=linewidth, linestyle=linestyleq, label='variational quasi-Newton')
plt.plot(iter_num, loss_mean[method, 5].T, linewidth=linewidth, linestyle=linestylef, label='first-order VI', color='k')
plt.title('Training Loss (VI)')
plt.xlabel('iteration number')
plt.ylim(losslims)
plt.gca().tick_params(axis='both', direction='in')
plt.legend()
if make_tikz:
    tikzplotlib.save('/Users/wilkinw1/postdoc/inprogress/newton-smoothers/paper/fig/product-vi-loss.tex',
                     axis_width='\\figurewidth',
                     axis_height='\\figureheight',
                     tex_relative_path_to_data='./fig/')

# plt.figure(4, figsize=(fig_width, fig_height))
plt.figure(5)
plt.plot(iter_num, nlpd_mean[method, 0].T, linewidth=linewidth, linestyle=linestyleh, label='heuristic VI')
plt.plot(iter_num, nlpd_mean[method, 3].T, linewidth=linewidth, linestyle=linestyler, label='Riemannian grads VI')
plt.plot(iter_num, nlpd_mean[method, 1].T, linewidth=linewidth, linestyle=linestyleg, label='variational Gauss-Newton')
plt.plot(iter_num, nlpd_mean[method, 2].T, linewidth=linewidth, linestyle=linestyleq, label='variational quasi-Newton')
plt.plot(iter_num, nlpd_mean[method, 5].T, linewidth=linewidth, linestyle=linestylef, label='first-order VI', color='k')
plt.title('Test NLPD (VI)')
plt.xlabel('iteration number')
plt.ylim(nlpdlims)
plt.gca().tick_params(axis='both', direction='in')
# plt.legend()
if make_tikz:
    tikzplotlib.save('/Users/wilkinw1/postdoc/inprogress/newton-smoothers/paper/fig/product-vi-nlpd.tex',
                     axis_width='\\figurewidth',
                     axis_height='\\figureheight',
                     tex_relative_path_to_data='./fig/')

plt.figure(6)
plt.plot(iter_num, rmse_mean[method, 0].T, linewidth=linewidth, linestyle=linestyleh, label='heuristic VI')
plt.plot(iter_num, rmse_mean[method, 3].T, linewidth=linewidth, linestyle=linestyler, label='Riemannian grads VI')
plt.plot(iter_num, rmse_mean[method, 1].T, linewidth=linewidth, linestyle=linestyleg, label='variational Gauss-Newton')
plt.plot(iter_num, rmse_mean[method, 2].T, linewidth=linewidth, linestyle=linestyleq, label='variational quasi-Newton')
plt.plot(iter_num, rmse_mean[method, 5].T, linewidth=linewidth, linestyle=linestylef, label='first-order VI', color='k')

plt.title('Ground Truth RMSE (VI)')
plt.xlabel('iteration number')
plt.ylim(rmselims)
plt.gca().tick_params(axis='both', direction='in')
# plt.legend()
if make_tikz:
    tikzplotlib.save('/Users/wilkinw1/postdoc/inprogress/newton-smoothers/paper/fig/product-vi-rmse.tex',
                     axis_width='\\figurewidth',
                     axis_height='\\figureheight',
                     tex_relative_path_to_data='./fig/')

method = 2

# plt.figure(5, figsize=(fig_width, fig_height))
plt.figure(7)
plt.plot(iter_num, loss_mean[method, 0].T, linewidth=linewidth, linestyle=linestyleh, label='heuristic PEP')
plt.plot(iter_num, loss_mean[method, 3].T, linewidth=linewidth, linestyle=linestyler, label='Riemannian grads PEP')
plt.plot(iter_num, loss_mean[method, 1].T, linewidth=linewidth, linestyle=linestyleg)
plt.plot(iter_num, loss_mean[method, 2].T, linewidth=linewidth, linestyle=linestyleq, label='PEP quasi-Newton')
plt.title('Training Loss (PEP)')
plt.xlabel('iteration number')
plt.ylim(losslims)
plt.gca().tick_params(axis='both', direction='in')
plt.legend()
if make_tikz:
    tikzplotlib.save('/Users/wilkinw1/postdoc/inprogress/newton-smoothers/paper/fig/product-pep-loss.tex',
                     axis_width='\\figurewidth',
                     axis_height='\\figureheight',
                     tex_relative_path_to_data='./fig/')

# plt.figure(6, figsize=(fig_width, fig_height))
plt.figure(8)
plt.plot(iter_num, nlpd_mean[method, 0].T, linewidth=linewidth, linestyle=linestyleh, label='heuristic PEP')
plt.plot(iter_num, nlpd_mean[method, 3].T, linewidth=linewidth, linestyle=linestyler, label='Riemannian grads PEP')
plt.plot(iter_num, nlpd_mean[method, 1].T, linewidth=linewidth, linestyle=linestyleg)
plt.plot(iter_num, nlpd_mean[method, 2].T, linewidth=linewidth, linestyle=linestyleq, label='PEP quasi-Newton')
plt.title('Test NLPD (PEP)')
plt.xlabel('iteration number')
plt.ylim(nlpdlims)
plt.gca().tick_params(axis='both', direction='in')
# plt.legend()
if make_tikz:
    tikzplotlib.save('/Users/wilkinw1/postdoc/inprogress/newton-smoothers/paper/fig/product-pep-nlpd.tex',
                     axis_width='\\figurewidth',
                     axis_height='\\figureheight',
                     tex_relative_path_to_data='./fig/')

plt.figure(9)
plt.plot(iter_num, rmse_mean[method, 0].T, linewidth=linewidth, linestyle=linestyleh, label='heuristic PEP')
plt.plot(iter_num, rmse_mean[method, 3].T, linewidth=linewidth, linestyle=linestyler, label='Riemannian grads PEP')
plt.plot(iter_num, rmse_mean[method, 1].T, linewidth=linewidth, linestyle=linestyleg)
plt.plot(iter_num, rmse_mean[method, 2].T, linewidth=linewidth, linestyle=linestyleq, label='PEP quasi-Newton')
plt.title('Ground Truth RMSE (PEP)')
plt.xlabel('iteration number')
plt.ylim(rmselims)
plt.gca().tick_params(axis='both', direction='in')
# plt.legend()
if make_tikz:
    tikzplotlib.save('/Users/wilkinw1/postdoc/inprogress/newton-smoothers/paper/fig/product-pep-rmse.tex',
                     axis_width='\\figurewidth',
                     axis_height='\\figureheight',
                     tex_relative_path_to_data='./fig/')

method = 3

plt.figure(10)
plt.plot(iter_num, loss_mean[method, 4].T, linewidth=linewidth, linestyle=linestyleh, label='PL')
plt.plot(iter_num, loss_mean[method, 3].T, linewidth=linewidth, linestyle=linestyler, label='Riemannian grads PL2')
plt.plot(iter_num, loss_mean[method, 1].T, linewidth=linewidth, linestyle=linestyleg, label='PL2 Gauss-Newton')
plt.plot(iter_num, loss_mean[method, 2].T, linewidth=linewidth, linestyle=linestyleq, label='PL2 quasi-Newton')
plt.plot(iter_num, loss_mean[method, 0].T, linewidth=linewidth, linestyle=linestyleh, label='heuristic PL2')
# plt.plot(iter_num, loss_mean[method, 5].T, linewidth=linewidth, linestyle=linestyleq, label='PL quasi-Newton', color='c')
plt.title('Training Loss (PL2)')
plt.xlabel('iteration number')
plt.ylim(losslims)
plt.gca().tick_params(axis='both', direction='in')
plt.legend()
if make_tikz:
    tikzplotlib.save('/Users/wilkinw1/postdoc/inprogress/newton-smoothers/paper/fig/product-pl2-loss.tex',
                     axis_width='\\figurewidth',
                     axis_height='\\figureheight',
                     tex_relative_path_to_data='./fig/')

plt.figure(11)
plt.plot(iter_num, nlpd_mean[method, 4].T, linewidth=linewidth, linestyle=linestyleh, label='PL')
plt.plot(iter_num, nlpd_mean[method, 3].T, linewidth=linewidth, linestyle=linestyler, label='Riemannian grads PL2')
plt.plot(iter_num, nlpd_mean[method, 1].T, linewidth=linewidth, linestyle=linestyleg, label='PL2 Gauss-Newton')
plt.plot(iter_num, nlpd_mean[method, 2].T, linewidth=linewidth, linestyle=linestyleq, label='PL2 quasi-Newton')
plt.plot(iter_num, nlpd_mean[method, 0].T, linewidth=linewidth, linestyle=linestyleh, label='heuristic PL2')
# plt.plot(iter_num, nlpd_mean[method, 5].T, linewidth=linewidth, linestyle=linestyleq, label='PL quasi-Newton', color='c')
plt.title('Test NLPD (PL2)')
plt.xlabel('iteration number')
plt.ylim(nlpdlims)
plt.gca().tick_params(axis='both', direction='in')
# plt.legend()
if make_tikz:
    tikzplotlib.save('/Users/wilkinw1/postdoc/inprogress/newton-smoothers/paper/fig/product-pl2-nlpd.tex',
                     axis_width='\\figurewidth',
                     axis_height='\\figureheight',
                     tex_relative_path_to_data='./fig/')

plt.figure(12)
plt.plot(iter_num, rmse_mean[method, 4].T, linewidth=linewidth, linestyle=linestyleh, label='PL')
plt.plot(iter_num, rmse_mean[method, 3].T, linewidth=linewidth, linestyle=linestyler, label='Riemannian grads PL2')
plt.plot(iter_num, rmse_mean[method, 1].T, linewidth=linewidth, linestyle=linestyleg, label='PL2 Gauss-Newton')
plt.plot(iter_num, rmse_mean[method, 2].T, linewidth=linewidth, linestyle=linestyleq, label='PL2 quasi-Newton')
plt.plot(iter_num, rmse_mean[method, 0].T, linewidth=linewidth, linestyle=linestyleh, label='heuristic PL2')
# plt.plot(iter_num, rmse_mean[method, 5].T, linewidth=linewidth, linestyle=linestyleq, label='PL quasi-Newton', color='c')
plt.title('Ground Truth RMSE (PL2)')
plt.xlabel('iteration number')
plt.ylim(rmselims)
plt.gca().tick_params(axis='both', direction='in')
# plt.legend()
if make_tikz:
    tikzplotlib.save('/Users/wilkinw1/postdoc/inprogress/newton-smoothers/paper/fig/product-pl2-rmse.tex',
                     axis_width='\\figurewidth',
                     axis_height='\\figureheight',
                     tex_relative_path_to_data='./fig/')

plt.show()
