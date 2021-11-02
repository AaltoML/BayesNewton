import pickle
import numpy as np

cpugpu = 'gpu'

if cpugpu == 'gpu':
    par = 1
else:
    par = 0

print('ST-SVGP')

ind_time = np.zeros([5])
for ind in range(5):
    with open("output/0_" + str(ind) + "_" + str(par) + "_" + str(cpugpu) + "_time.txt", "rb") as fp:
        ind_time[ind] = pickle.load(fp)

ind_rmse = np.zeros([5])
for ind in range(5):
    with open("output/0_" + str(ind) + "_" + str(par) + "_" + str(cpugpu) + "_rmse.txt", "rb") as fp:
        ind_rmse[ind] = pickle.load(fp)

ind_nlpd = np.zeros([5])
for ind in range(5):
    with open("output/0_" + str(ind) + "_" + str(par) + "_" + str(cpugpu) + "_nlpd.txt", "rb") as fp:
        ind_nlpd[ind] = pickle.load(fp)

print('time:')
np.set_printoptions(precision=3)
print(np.mean(ind_time))
np.set_printoptions(precision=2)
print(np.std(ind_time))

print('rmse:')
np.set_printoptions(precision=3)
print(np.mean(ind_rmse))
np.set_printoptions(precision=2)
print(np.std(ind_rmse))

print('nlpd:')
np.set_printoptions(precision=3)
print(np.mean(ind_nlpd))
np.set_printoptions(precision=2)
print(np.std(ind_nlpd))


print('MF-ST-SVGP')

ind_time = np.zeros([5])
for ind in range(5):
    with open("output/1_" + str(ind) + "_" + str(par) + "_" + str(cpugpu) + "_time.txt", "rb") as fp:
        ind_time[ind] = pickle.load(fp)

ind_rmse = np.zeros([5])
for ind in range(5):
    with open("output/1_" + str(ind) + "_" + str(par) + "_" + str(cpugpu) + "_rmse.txt", "rb") as fp:
        ind_rmse[ind] = pickle.load(fp)

ind_nlpd = np.zeros([5])
for ind in range(5):
    with open("output/1_" + str(ind) + "_" + str(par) + "_" + str(cpugpu) + "_nlpd.txt", "rb") as fp:
        ind_nlpd[ind] = pickle.load(fp)

print('time:')
np.set_printoptions(precision=3)
print(np.mean(ind_time))
np.set_printoptions(precision=2)
print(np.std(ind_time))

print('rmse:')
np.set_printoptions(precision=3)
print(np.mean(ind_rmse))
np.set_printoptions(precision=2)
print(np.std(ind_rmse))

print('nlpd:')
np.set_printoptions(precision=3)
print(np.mean(ind_nlpd))
np.set_printoptions(precision=2)
print(np.std(ind_nlpd))


print('SVGP0')

ind_time = np.zeros([5])
for ind in range(5):
    with open("output/gpflow_" + str(ind) + "_0_" + str(cpugpu) + "_time.txt", "rb") as fp:
        ind_time[ind] = pickle.load(fp)

ind_rmse = np.zeros([5])
for ind in range(5):
    with open("output/gpflow_" + str(ind) + "_0_" + str(cpugpu) + "_rmse.txt", "rb") as fp:
        ind_rmse[ind] = pickle.load(fp)

ind_nlpd = np.zeros([5])
for ind in range(5):
    with open("output/gpflow_" + str(ind) + "_0_" + str(cpugpu) + "_nlpd.txt", "rb") as fp:
        ind_nlpd[ind] = pickle.load(fp)

print('time:')
np.set_printoptions(precision=3)
print(np.mean(ind_time))
np.set_printoptions(precision=2)
print(np.std(ind_time))

print('rmse:')
np.set_printoptions(precision=3)
print(np.mean(ind_rmse))
np.set_printoptions(precision=2)
print(np.std(ind_rmse))

print('nlpd:')
np.set_printoptions(precision=3)
print(np.mean(ind_nlpd))
np.set_printoptions(precision=2)
print(np.std(ind_nlpd))


print('SVGP1')

ind_time = np.zeros([5])
for ind in range(5):
    with open("output/gpflow_" + str(ind) + "_1_" + str(cpugpu) + "_time.txt", "rb") as fp:
        ind_time[ind] = pickle.load(fp)

ind_rmse = np.zeros([5])
for ind in range(5):
    with open("output/gpflow_" + str(ind) + "_1_" + str(cpugpu) + "_rmse.txt", "rb") as fp:
        ind_rmse[ind] = pickle.load(fp)

ind_nlpd = np.zeros([5])
for ind in range(5):
    with open("output/gpflow_" + str(ind) + "_1_" + str(cpugpu) + "_nlpd.txt", "rb") as fp:
        ind_nlpd[ind] = pickle.load(fp)

print('time:')
np.set_printoptions(precision=3)
print(np.mean(ind_time))
np.set_printoptions(precision=2)
print(np.std(ind_time))

print('rmse:')
np.set_printoptions(precision=3)
print(np.mean(ind_rmse))
np.set_printoptions(precision=2)
print(np.std(ind_rmse))

print('nlpd:')
np.set_printoptions(precision=3)
print(np.mean(ind_nlpd))
np.set_printoptions(precision=2)
print(np.std(ind_nlpd))


print('SVGP2')

ind_time = np.zeros([5])
for ind in range(5):
    with open("output/gpflow_" + str(ind) + "_2_" + str(cpugpu) + "_time.txt", "rb") as fp:
        ind_time[ind] = pickle.load(fp)

ind_rmse = np.zeros([5])
for ind in range(5):
    with open("output/gpflow_" + str(ind) + "_2_" + str(cpugpu) + "_rmse.txt", "rb") as fp:
        ind_rmse[ind] = pickle.load(fp)

ind_nlpd = np.zeros([5])
for ind in range(5):
    with open("output/gpflow_" + str(ind) + "_2_" + str(cpugpu) + "_nlpd.txt", "rb") as fp:
        ind_nlpd[ind] = pickle.load(fp)

print('time:')
np.set_printoptions(precision=3)
print(np.mean(ind_time))
np.set_printoptions(precision=2)
print(np.std(ind_time))

print('rmse:')
np.set_printoptions(precision=3)
print(np.mean(ind_rmse))
np.set_printoptions(precision=2)
print(np.std(ind_rmse))

print('nlpd:')
np.set_printoptions(precision=3)
print(np.mean(ind_nlpd))
np.set_printoptions(precision=2)
print(np.std(ind_nlpd))


if cpugpu == 'gpu':

    print('SVGP3')

    ind_time = np.zeros([5])
    for ind in range(5):
        with open("output/gpflow_" + str(ind) + "_3_" + str(cpugpu) + "_time.txt", "rb") as fp:
            ind_time[ind] = pickle.load(fp)

    ind_rmse = np.zeros([5])
    for ind in range(5):
        with open("output/gpflow_" + str(ind) + "_3_" + str(cpugpu) + "_rmse.txt", "rb") as fp:
            ind_rmse[ind] = pickle.load(fp)

    ind_nlpd = np.zeros([5])
    for ind in range(5):
        with open("output/gpflow_" + str(ind) + "_3_" + str(cpugpu) + "_nlpd.txt", "rb") as fp:
            ind_nlpd[ind] = pickle.load(fp)

    print('time:')
    np.set_printoptions(precision=3)
    print(np.mean(ind_time))
    np.set_printoptions(precision=2)
    print(np.std(ind_time))

    print('rmse:')
    np.set_printoptions(precision=3)
    print(np.mean(ind_rmse))
    np.set_printoptions(precision=2)
    print(np.std(ind_rmse))

    print('nlpd:')
    np.set_printoptions(precision=3)
    print(np.mean(ind_nlpd))
    np.set_printoptions(precision=2)
    print(np.std(ind_nlpd))


    print('SVGP4')

    ind_time = np.zeros([5])
    for ind in range(5):
        with open("output/gpflow_" + str(ind) + "_4_" + str(cpugpu) + "_time.txt", "rb") as fp:
            ind_time[ind] = pickle.load(fp)

    ind_rmse = np.zeros([5])
    for ind in range(5):
        with open("output/gpflow_" + str(ind) + "_4_" + str(cpugpu) + "_rmse.txt", "rb") as fp:
            ind_rmse[ind] = pickle.load(fp)

    ind_nlpd = np.zeros([5])
    for ind in range(5):
        with open("output/gpflow_" + str(ind) + "_4_" + str(cpugpu) + "_nlpd.txt", "rb") as fp:
            ind_nlpd[ind] = pickle.load(fp)

    print('time:')
    np.set_printoptions(precision=3)
    print(np.mean(ind_time))
    np.set_printoptions(precision=2)
    print(np.std(ind_time))

    print('rmse:')
    np.set_printoptions(precision=3)
    print(np.mean(ind_rmse))
    np.set_printoptions(precision=2)
    print(np.std(ind_rmse))

    print('nlpd:')
    np.set_printoptions(precision=3)
    print(np.mean(ind_nlpd))
    np.set_printoptions(precision=2)
    print(np.std(ind_nlpd))


print('SKI')

ind_time = np.zeros([5])
for ind in range(5):
    with open("output/ski_" + str(ind) + "_" + str(par) + "_time.txt", "rb") as fp:
        ind_time[ind] = pickle.load(fp)

ind_rmse = np.zeros([5])
for ind in range(5):
    with open("output/ski_" + str(ind) + "_" + str(par) + "_rmse.txt", "rb") as fp:
        ind_rmse[ind] = pickle.load(fp)

ind_nlpd = np.zeros([5])
for ind in range(5):
    with open("output/ski_" + str(ind) + "_" + str(par) + "_nlpd.txt", "rb") as fp:
        ind_nlpd[ind] = pickle.load(fp)

print('time:')
np.set_printoptions(precision=3)
print(np.mean(ind_time))
np.set_printoptions(precision=2)
print(np.std(ind_time))

print('rmse:')
np.set_printoptions(precision=3)
print(np.mean(ind_rmse))
np.set_printoptions(precision=2)
print(np.std(ind_rmse))

print('nlpd:')
np.set_printoptions(precision=3)
print(np.mean(ind_nlpd))
np.set_printoptions(precision=2)
print(np.std(ind_nlpd))
