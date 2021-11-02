import pickle
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

color_palette = {
    'black': '#000000',
    'orange': '#E69F00',
    'blue': '#56B4E9',
    'green': '#009E73',
    'orange': '#F0E442',
    'dark_blue': '#0072B2',
    'dark_orange': '#D55E00',
    'pink': '#CC79A7',
    'white': '#111111',
    'grey': 'grey'
}

# timings = np.zeros([3, 10])
num_complete = 7
timings = np.zeros([3, num_complete])
for model_type in range(3):
    # for nr_ind in range(10):
    for nr_ind in range(num_complete):
        with open("output/cpu_" + str(model_type) + "_" + str(nr_ind) + "_time.txt", "rb") as fp:
            result = pickle.load(fp)
            # print(result)
            timings[model_type, nr_ind] = result

num_complete_nlpd = 7
nlpd = np.zeros([3, num_complete_nlpd])
for model_type in range(3):
    # for nr_ind in range(10):
    for nr_ind in range(num_complete_nlpd):
        nlpdf = 0.
        for fold in range(10):
            print("output/" + str(model_type) + "_" + str(nr_ind) + "_" + str(fold) + "_nlpd.txt")
            with open("output/" + str(model_type) + "_" + str(nr_ind) + "_" + str(fold) + "_nlpd.txt", "rb") as fp:
                result = pickle.load(fp)
            nlpdf += result
        # print(result)
        nlpd[model_type, nr_ind] = nlpdf / 10

num_space = np.array([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
plt.figure(1)
plt.plot(num_space[:num_complete], timings.T[:, 0], '.--', markersize=3, linewidth=2.5, color=color_palette['dark_orange'])
plt.plot(num_space[:num_complete], timings.T[:, 1], 'x--', markersize=5, linewidth=2.5, color=color_palette['dark_blue'])
plt.plot(num_space[:num_complete], timings.T[:, 2], 'x-.', markersize=5, linewidth=2.5, color=color_palette['green'])
plt.xlabel('Number of spatial points')
plt.ylabel('Training step time (secs)')
plt.ylim([-0., 82])
ax = plt.gca()
ax.set_xticks(num_space[:num_complete])
plt.legend(['Full', 'Spatial mean-field', 'Infinite-horizon'], loc=2)
if True:
    tikzplotlib.save('//Users/wilkinw1/postdoc/inprogress/ati-fcai/paper/icml2021/fig/scalability.tex',
                     axis_width='\\figurewidth',
                     axis_height='\\figureheight',
                     tex_relative_path_to_data='./fig/')

plt.figure(2)
plt.plot(num_space[:num_complete_nlpd], nlpd.T[:, 0], '.--', markersize=3, linewidth=2.5, color=color_palette['dark_orange'])
plt.plot(num_space[:num_complete_nlpd], nlpd.T[:, 1], 'x--', markersize=5, linewidth=2.5, color=color_palette['dark_blue'])
plt.plot(num_space[:num_complete_nlpd], nlpd.T[:, 2], 'x-.', markersize=6, linewidth=2.5, color=color_palette['green'])
plt.plot(num_space[:num_complete_nlpd], nlpd.T[:, 0], '.--', markersize=3, linewidth=2.5, color=color_palette['dark_orange'])
plt.xlabel('Number of spatial points')
plt.ylabel('Test NLPD')
# plt.ylim([-0., 82])
ax = plt.gca()
ax.set_xticks(num_space[:num_complete_nlpd])
plt.legend(['Full', 'Spatial mean-field', 'Infinite-horizon'], loc=1)
if True:
    tikzplotlib.save('//Users/wilkinw1/postdoc/inprogress/ati-fcai/paper/icml2021/fig/approx_perf.tex',
                     axis_width='\\figurewidth',
                     axis_height='\\figureheight',
                     tex_relative_path_to_data='./fig/')
plt.show()
