from matplotlib import rc

import matplotlib.pyplot as plt
import numpy as np
import pickle

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.style.use('ggplot')

rc('text', usetex=False)

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


time_steps = [200, 1000, 2000, 3000]
cpu_times = []
gpu_times = []
parallel_times = []
for i in range(4):
    with open('output/0_0_{0}_0_gpu.txt'.format(i), 'rb') as f:
        gpu_times.append(pickle.load(f))

    with open('output/0_0_{0}_0_cpu.txt'.format(i), 'rb') as f:
        cpu_times.append(pickle.load(f))

    with open('output/0_0_{0}_1_gpu.txt'.format(i), 'rb') as f:
        parallel_times.append(pickle.load(f))

gpu_times = np.array(gpu_times)
cpu_times = np.array(cpu_times)
parallel_times = np.array(parallel_times)

plt.figure()
plt.plot(time_steps, gpu_times, 'k-o', label='GPU (sequential)')
plt.plot(time_steps, cpu_times, 'r-o', label='CPU (sequential)')
plt.plot(time_steps, parallel_times, 'k--o', label='GPU (parallel)')
plt.ylabel('iteration time')
plt.xlabel('time steps')
plt.legend()
plt.savefig('output/timings.pdf', bbox_inches='tight')

plt.figure()
plt.plot(time_steps, cpu_times, 'r-o', label='CPU (sequential)')
plt.plot(time_steps, parallel_times, 'k--o', label='GPU (parallel)')
plt.ylabel('iteration time')
plt.xlabel('time steps')
plt.legend()
plt.savefig('output/timings_zoom.pdf', bbox_inches='tight')

# plt.figure()
# plt.plot(time_steps, gpu_times/gpu_times[0], 'k-o', label='GPU')
# plt.plot(time_steps, cpu_times/cpu_times[0], 'r-o', label='CPU')
# plt.ylabel('wall-time/(max(wall-time))')
# plt.xlabel('$spatial points$')
# plt.legend()
# plt.savefig('timings_normalised.pdf', bbox_inches='tight')



