'''Graph Convergence Data

'''

# import json
import pickle
import matplotlib.pyplot as plt
import numpy as np




# Resnet 1 - learning rate tests
#files = ['resnet_1.txt']

# Resnet 2 - elu tests
#files = ['res_elu_data1.txt',  'res_elu_data2.txt']
files = ['res_aug.txt', 'res_aug2.txt']



data = {
        'top1_t': [],
        'top5_t': [],
        'top1_v': [],
        'top5_v': [],
        'learning_rate': []
}

for file in files:
	tmp_data = {}
	with open(file, 'r') as f:
		tmp_data = pickle.load(f)

	if 'top5' in tmp_data:
		tmp_data['top5_v'] = tmp_data['top5']
		tmp_data['top1_v'] = tmp_data['top1']
		tmp_data['top1_t'] = []
		tmp_data['top5_t'] = []
	for v in tmp_data['top1_v']:
		data['top1_v'].append(v)
	for v in tmp_data['top5_v']:
		data['top5_v'].append(v)
	for v in tmp_data['top1_t']:
		data['top1_t'].append(v)
	for v in tmp_data['top5_t']:
		data['top5_t'].append(v)
	data['learning_rate'].append(tmp_data['learning_rate'])



#data = {}
#with open(filename, 'r') as file:
#	data = pickle.load(file)

batch_size = 250
step_display = 50
n = np.arange(len(data['top1_v']))
n = n*step_display

#plt.plot(n, data['top1_t'], label='t1')
#plt.plot(n, data['top5_t'], label='t5')
plt.plot(n, data['top5_v'], label='Top 5')
plt.plot(n, data['top1_v'], label='Top 1')

plt.title('Validation Accuracy vs Iteration')
plt.xlabel('Iterations')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()
