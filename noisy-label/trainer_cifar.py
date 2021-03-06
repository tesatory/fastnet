from fastnet import parser
import fastnet.net
from fastnet.layer import TRAIN, TEST
from fastnet import util
import numpy as np
from pycuda import gpuarray
import cudaconv2
import sys
import scipy.io

def copy_to_gpu(data):
    return gpuarray.to_gpu(data.astype(np.float32))

class BatchData(object):
    def __init__(self, data, labels):
        self.data = copy_to_gpu(data)
        self.labels = copy_to_gpu(labels)
        self.labels = self.labels.reshape((self.labels.size, 1))

def load_cifar100():
    base_dir = '/home/sainbar/data/cifar-100-python/'
    train_file = util.load(base_dir + 'train')
    train_data = train_file['data']
    train_data = train_data.T.copy()
    train_data = train_data.astype(np.float32)

    test_file = util.load(base_dir + 'test')
    test_data = test_file['data']
    test_data = test_data.T.copy()
    test_data = test_data.astype(np.float32)

    train_labels = np.asarray(train_file['fine_labels'], np.float32)
    test_labels = np.asarray(test_file['fine_labels'], np.float32)

    return train_data, train_labels, test_data, test_labels

# setting
batch_size = 128
param_file = '/home/sainbar/fastnet-self-paced/config/cifar-100.cfg'
num_epoch = 500
learning_rate = 1
image_color = 3
image_size = 32
image_shape = (image_color, image_size, image_size, batch_size)
init_model = parser.parse_config_file(param_file)

# prepare data
train_data, train_labels, test_data, test_labels = load_cifar100()
data_mean = train_data.mean(axis=1,keepdims=True)
train_data = train_data - data_mean
test_data = test_data - data_mean

train_size = train_data.shape[1]
test_size = test_data.shape[1]
train_batches = list()
test_batches = list()
ind = 0
while ind + batch_size <= train_size:
    batch = BatchData(train_data[:,ind:ind+batch_size], \
                          train_labels[ind:ind+batch_size])
    train_batches.append(batch)
    ind += batch_size

ind = 0
while ind + batch_size <= test_size:
    batch = BatchData(test_data[:,ind:ind+batch_size], \
                          test_labels[ind:ind+batch_size])
    test_batches.append(batch)
    ind += batch_size

print 'train:', train_data.shape[1], 'samples', len(train_batches), 'batches'
print 'test:', test_data.shape[1], 'samples', len(test_batches), 'batches'

net = fastnet.net.FastNet(learning_rate, image_shape, init_model)
for epoch in range(num_epoch):
    total_cases = 0
    total_correct = 0
    for batch in train_batches:
        net.train_batch(batch.data, batch.labels, TRAIN)
        cost, correct, num_case = net.get_batch_information()
        total_cases += num_case
        total_correct += correct * num_case
    train_error = (1. - 1.0*total_correct/total_cases)

    total_cases = 0
    total_correct = 0
    for batch in test_batches:
        net.train_batch(batch.data, batch.labels, TEST)
        cost, correct, num_case = net.get_batch_information()
        total_cases += num_case
        total_correct += correct * num_case
    test_error = (1. - 1.0*total_correct/total_cases)

    print 'epoch:', epoch, 'train-error:', train_error, \
        'test-error:', test_error
