import pickle
import numpy as np

empty_data = np.nan * np.zeros([20, 3])

for method in range(6):
    for fold in range(10):
        with open("output/varyM" + str(method) + "_" + str(fold) + ".txt", "wb") as fp:
            pickle.dump(empty_data, fp)
