from fastnet import parser
import fastnet.net
import numpy as np
import cudaconv2
import sys
import scipy.io
import net_trainer
import data_loader
from fastnet.layer import TRAIN, TEST
import data_selection

# setting
batch_size = 128
param_file = '/home/sainbar/fastnet/config/cifar-10-18pct.cfg'
num_epoch = 200
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

# noisy data
noisy_data, noisy_labels = data_loader.load_noisy_labeled()
noisy_data = noisy_data - data_mean
noisy_data = noisy_data[:,0:200000]
noisy_labels = noisy_labels[0:200000]

train_data = np.concatenate((train_data, noisy_data), axis=1)
train_labels = np.concatenate((train_labels, noisy_labels))
N = int(np.floor(train_data.shape[1] / batch_size) * batch_size)
train_data = train_data[:,0:N]
train_labels = train_labels[0:N]
train_batches = data_loader.prepare_batches(train_data, train_labels, batch_size)
test_batches = data_loader.prepare_batches(test_data, test_labels, batch_size)

M = 20000
while M < 50000:
	train_batches2 = data_selection.get_new_batches(train_batches, train_data, train_labels, M)
	net_trainer.train(net, 10, train_batches2, test_batches)
	M += 2000
