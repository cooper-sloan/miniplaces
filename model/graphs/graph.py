'''Graph Convergence Data

'''

# import json
import pickle
import matplotlib.pyplot as plt
import numpy as np

filename = 'conergence_data1.txt'


data = {}
with open(filename, 'r') as file:
	data = pickle.load(file)

batch_size = 250
step_display = 50
n = np.arange(len(data['top1_v']))
n = n*step_display

#plt.plot(n, data['top1_t'], label='t1')
#plt.plot(n, data['top5_t'], label='t5')
plt.plot(n, data['top1_v'], label='v1')
plt.plot(n, data['top5_v'], label='v5')
plt.legend()
plt.show()
