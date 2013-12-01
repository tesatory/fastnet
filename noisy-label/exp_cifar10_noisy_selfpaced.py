from fastnet import parser
import fastnet.net
import numpy as np
import cudaconv2
import sys
import scipy.io
import net_trainer
import data_loader
import data_selection

# setting
batch_size = 128
param_file = '/home/sainbar/fastnet/config/cifar-10-18pct.cfg'
num_epoch = 60
learning_rate = 1
image_color = 3
image_size = 32
image_shape = (image_color, image_size, image_size, batch_size)
init_model = parser.parse_config_file(param_file)
net = fastnet.net.FastNet(learning_rate, image_shape, init_model)

# prepare data
train_data, train_labels, test_data, test_labels = data_loader.load_cifar10()
data_mean = train_data.mean(axis=1,keepdims=True)
train_data = train_data - data_mean
test_data = test_data - data_mean

noise_sz = int(sys.argv[1])
if noise_sz > 0:
	# noisy data
	noisy_data, noisy_labels = data_loader.load_noisy_labeled()
	noisy_data = noisy_data - data_mean
	noisy_data = noisy_data[:,0:noise_sz]
	noisy_labels = noisy_labels[0:noise_sz]
	train_data = np.concatenate((train_data, noisy_data), axis=1)
	train_labels = np.concatenate((train_labels, noisy_labels))

order = range(train_data.shape[1])
np.random.shuffle(order)
train_data = train_data[:,order]
train_labels = train_labels[order]
N = int(np.floor(train_data.shape[1] / batch_size) * batch_size)
train_data = train_data[:,0:N]
train_labels = train_labels[0:N]

train_batches = data_loader.prepare_batches(train_data, train_labels, batch_size)
test_batches = data_loader.prepare_batches(test_data, test_labels, batch_size)
print '# train:', train_data.shape[1], 'samples', len(train_batches), 'batches'
print '# test:', test_data.shape[1], 'samples', len(test_batches), 'batches'
net_trainer.train(net, 10, train_batches, test_batches)

for k in range(5):
	train_batches2 = data_selection.get_new_batches(net, batch_size,train_batches, train_data, train_labels, 100000)
	net_trainer.train(net, 10, train_batches2, test_batches)
