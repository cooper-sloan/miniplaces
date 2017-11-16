'''Graph Convergence Data

'''

# import json
import pickle
import matplotlib.pyplot as plt
import numpy as np

filename = 'conergence_data.txt'


data = {}
with open(filename, 'r') as file:
	data = pickle.load(file)


n = np.arange(len(data['top1']))

plt.plot(n, data['top1'])

plt.show()