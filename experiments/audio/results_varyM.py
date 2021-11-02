import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes, mark_inset
import tikzplotlib
from matplotlib._png import read_png

save_tikz = True

method_nlml = np.zeros([6, 10, 20])
method_nlpd = np.zeros([6, 10, 20])
method_rmse = np.zeros([6, 10, 20])
for method in range(6):
    for dataset in range(10):
        with open("output/varyM" + str(method) + "_" + str(dataset) + ".txt", "rb") as fp:
            file = pickle.load(fp)
            nlml = np.array(file[:, 0])
            nlpd = np.array(file[:, 1])
            rmse = np.array(file[:, 2])
            method_nlml[method, dataset] = nlml
            method_nlpd[method, dataset] = nlpd
            method_rmse[method, dataset] = rmse
# svgp_nlml = np.zeros([10, 50])
# svgp_nlpd = np.zeros([10, 50])
# svgp_classerror = np.zeros([10, 50])
# for dataset in range(10):
#     with open("baselines/output/varyM" + str(dataset) + ".txt", "rb") as fp:
#         file = pickle.load(fp)
#         nlml = np.array(file[:, 0])
#         nlpd = np.array(file[:, 1])
#         classerror = np.array(file[:, 2])
#         svgp_nlml[dataset] = nlml
#         svgp_nlpd[dataset] = nlpd
#         svgp_classerror[dataset] = classerror

# np.set_printoptions(precision=3)
# print(np.mean(method_nlml, axis=1))
# np.set_printoptions(precision=2)
# print(np.nanstd(method_nlml, axis=1))
lb_nlml = np.mean(method_nlml, axis=1) - 1 * np.std(method_nlml, axis=1)
ub_nlml = np.mean(method_nlml, axis=1) + 1 * np.std(method_nlml, axis=1)
# lb_nlml_svgp = np.mean(svgp_nlml, axis=0) - 0.1 * np.std(svgp_nlml, axis=0)
# ub_nlml_svgp = np.mean(svgp_nlml, axis=0) + 0.1 * np.std(svgp_nlml, axis=0)

# np.set_printoptions(precision=3)
# print(np.nanmean(method_nlpd, axis=1))
# np.set_printoptions(precision=2)
# print(np.nanstd(method_nlpd, axis=1))
lb_nlpd = np.mean(method_nlpd, axis=1) - 1 * np.std(method_nlpd, axis=1)
ub_nlpd = np.mean(method_nlpd, axis=1) + 1 * np.std(method_nlpd, axis=1)
# lb_nlpd_svgp = np.mean(svgp_nlpd, axis=0) - 0.1 * np.std(svgp_nlpd, axis=0)
# ub_nlpd_svgp = np.mean(svgp_nlpd, axis=0) + 0.1 * np.std(svgp_nlpd, axis=0)

lb_classerror = np.mean(method_rmse, axis=1) - 1 * np.std(method_rmse, axis=1)
ub_classerror = np.mean(method_rmse, axis=1) + 1 * np.std(method_rmse, axis=1)
# lb_classerror_svgp = np.mean(svgp_classerror, axis=0) - 0.1 * np.std(svgp_classerror, axis=0)
# ub_classerror_svgp = np.mean(svgp_classerror, axis=0) + 0.1 * np.std(svgp_classerror, axis=0)

legend_entries = ['S$^2$EKS', 'S$^2$PL', 'S$^2$PEP($\\alpha=1$)', 'S$^2$PEP($\\alpha=0.5$)', 'S$^2$PEP($\\alpha=0.01$)', 'S$^2$CVI']

num_inducing = np.linspace(100, 2000, 20, dtype=int)
fig0, ax0 = plt.subplots()
for method in [0, 1, 2, 3, 4, 5]:
    ax0.plot(num_inducing, np.mean(method_nlml, axis=1)[method].T, label=legend_entries[method], linewidth=2)
    ax0.fill_between(num_inducing, lb_nlml[method], ub_nlml[method], alpha=0.05)
# plt.plot(num_inducing, np.mean(svgp_nlml, axis=0).T, label='VI (sparse)')
# plt.fill_between(num_inducing, lb_nlml_svgp, ub_nlml_svgp, alpha=0.05)
# plt.legend(loc=1)
# plt.ylabel('NLML')
plt.legend(loc=3)
plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    direction='in')
plt.xlabel('Number of inducing inputs, M')
plt.xlim(0, 2000)
plt.ylim(0, 30000)
plt.yticks([0, 10000, 20000])

# yl = ax0.get_ylim()
# x1, x2, y1, y2 = 100, 700, 18850, 43000  # specify the limits
# ypos = yl[0]+(yl[1]-yl[0])/1.7
# print(ypos)
# ax0.text(1200, ypos, '\\tikz\\node[coordinate] (n1) {};')
# ax0.text(x1, y2, '\\tikz\\node[coordinate] (r1) {};')
# ax0.text(x2, y1, '\\tikz\\node[coordinate] (r2) {};')

if save_tikz:
    tikzplotlib.save('audio_nlml.tex',
                     axis_width='\\figurewidth',
                     axis_height='\\figureheight',
                     tex_relative_path_to_data='./graphs/')

# fig1, ax1 = plt.subplots()
# for method in [0, 1, 2, 3, 4, 5]:
#     ax1.plot(num_inducing, np.mean(method_nlml, axis=1)[method].T, label=legend_entries[method], linewidth=2)
#     ax1.fill_between(num_inducing, lb_nlml[method], ub_nlml[method], alpha=0.05)
# ax1.set_xlim(x1, x2)
# ax1.set_ylim(y1, y2)
# ax1.set_xticks([])
# ax1.set_yticks([])
# ax1.text(x1, y2, '\\tikz\\node[coordinate] (z1) {};')
# ax1.text(x2, y1, '\\tikz\\node[coordinate] (z2) {};')
#
# if save_tikz:
#     tikzplotlib.save('audio_nlml_zoom.tex',
#                      axis_width='\\figurewidth',
#                      axis_height='\\figureheight',
#                      tex_relative_path_to_data='./graphs/')


fig2, ax2 = plt.subplots()
for method in [0, 1, 2, 3, 4, 5]:
    ax2.plot(num_inducing, np.mean(method_nlpd, axis=1)[method].T, label=legend_entries[method], linewidth=2)
    ax2.fill_between(num_inducing, lb_nlpd[method], ub_nlpd[method], alpha=0.05)
# plt.plot(num_inducing, np.mean(svgp_nlpd, axis=0).T, label='VI (sparse)')
# plt.fill_between(num_inducing, lb_nlpd_svgp, ub_nlpd_svgp, alpha=0.05)
# ax2.legend(loc=1)
# plt.ylabel('NLPD')
plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    direction='in')
plt.xlabel('Number of inducing inputs, M')
ax2.set_xlim(0, 2000)
ax2.set_ylim(-0.8, 1.25)

# yl = ax2.get_ylim()
# x1, x2, y1, y2 = 100, 640, 0.69, 1.75  # specify the limits
# ypos = yl[0]+(yl[1]-yl[0])/1.5
# ax2.text(1200, ypos, '\\tikz\\node[coordinate] (n2) {};')
# ax2.text(x1, y2, '\\tikz\\node[coordinate] (r3) {};')
# ax2.text(x2, y1, '\\tikz\\node[coordinate] (r4) {};')

if save_tikz:
    tikzplotlib.save('audio_nlpd.tex',
                     axis_width='\\figurewidth',
                     axis_height='\\figureheight',
                     tex_relative_path_to_data='./graphs/')

# fig3, ax3 = plt.subplots()
# for method in [0, 1, 2, 3, 4, 5]:
#     ax3.plot(num_inducing, np.mean(method_nlpd, axis=1)[method].T, label=legend_entries[method], linewidth=2)
#     ax3.fill_between(num_inducing, lb_nlpd[method], ub_nlpd[method], alpha=0.05)
# # plt.plot(num_inducing, np.mean(svgp_nlml, axis=0).T, label='VI (sparse)')
# # plt.fill_between(num_inducing, lb_nlml_svgp, ub_nlml_svgp, alpha=0.05)
# # plt.ylabel('NLPD')
# # plt.xlabel('Number of inducing inputs, M')
# ax3.set_xlim(x1, x2)
# ax3.set_ylim(y1, y2)
# ax3.set_xticks([])
# ax3.set_yticks([])
# ax3.text(x1, y2, '\\tikz\\node[coordinate] (z3) {};')
# ax3.text(x2, y1, '\\tikz\\node[coordinate] (z4) {};')
#
# if save_tikz:
#     tikzplotlib.save('audio_nlpd_zoom.tex',
#                      axis_width='\\figurewidth',
#                      axis_height='\\figureheight',
#                      tex_relative_path_to_data='./graphs/')

fig4, ax4 = plt.subplots()
for method in [0, 1, 2, 3, 4, 5]:
    ax4.plot(num_inducing, np.mean(method_rmse, axis=1)[method].T, label=legend_entries[method], linewidth=2)
    ax4.fill_between(num_inducing, lb_classerror[method], ub_classerror[method], alpha=0.05)
# ax4.plot(num_inducing, np.mean(svgp_classerror, axis=0).T, label='VI (sparse)')
# ax4.fill_between(num_inducing, lb_classerror_svgp, ub_classerror_svgp, alpha=0.05)
# plt.legend(loc=9)
# plt.ylabel('Classification Error')
plt.xlabel('Number of inducing inputs, M')
plt.xlim(0, 2000)
# plt.ylim(0.0235, 0.05)
plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    direction='in')

if save_tikz:
    tikzplotlib.save('audio_rmse.tex',
                     axis_width='\\figurewidth',
                     axis_height='\\figureheight',
                     tex_relative_path_to_data='./graphs/')

plt.show()
