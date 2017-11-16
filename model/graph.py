'''Graph Convergence Data

'''

# import json
import pickle
import matplotlib.pyplot as plt
import numpy as np

filename = 'convergence_data.txt'


data = {}
with open(filename, 'r') as file:
	data = pickle.load(file)


n = arrange(len(data['top1']))

plt.plot(n, data['top1'])

