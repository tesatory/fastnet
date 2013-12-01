from fastnet import parser
import fastnet.net
from fastnet.layer import TRAIN, TEST
from fastnet import util
import numpy as np
import pycuda.driver as cuda
from pycuda import gpuarray
import cudaconv2
import sys
import scipy.io

def copy_to_gpu(data):
    return gpuarray.to_gpu(data.astype(np.float32))

def copy_to_cpu(gpu_data):
    data = np.zeros(gpu_data.shape).astype(np.float32)
    cuda.memcpy_dtoh(data, gpu_data.gpudata)
    return data
    
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

def load_noise():
    data_file = scipy.io.loadmat('/home/sainbar/data/cifar-100/tiny_img_noise.mat')
    data = data_file['data']
    data = data.astype(np.float32)
    data = data.copy()
    return data

def prepare_batches(train_data, train_labels, test_data, test_labels, batch_size):
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
    return train_batches, test_batches

def train(net, num_epoch, train_batches, test_batches):
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

# setting
batch_size = 128
param_file = '/home/sainbar/fastnet-self-paced/config/cifar-100.cfg'
num_epoch = 10
num_epoch2 = 80
learning_rate = 1
image_color = 3
image_size = 32
image_shape = (image_color, image_size, image_size, batch_size)
init_model = parser.parse_config_file(param_file)
net = fastnet.net.FastNet(learning_rate, image_shape, init_model)

# prepare data
train_data, train_labels, test_data, test_labels = load_cifar100()
data_mean = train_data.mean(axis=1,keepdims=True)
train_data = train_data - data_mean
test_data = test_data - data_mean

# noise data
noise_sz = int(sys.argv[1])
noise_data = load_noise()
noise_data = noise_data - data_mean
noise_labels = np.zeros(noise_data.shape[1]).astype(np.float32)
for i in range(len(noise_labels)):
    noise_labels[i] = np.random.randint(100)

print "add noise", noise_sz
is_noise = np.concatenate((np.zeros(train_labels.shape), np.ones(noise_sz)))
train_data = np.concatenate((train_data, noise_data[:,:noise_sz]), axis=1)
train_labels = np.concatenate((train_labels, noise_labels[:noise_sz]))
N = int(np.floor(train_data.shape[1] / batch_size) * batch_size)
order = range(N)
np.random.shuffle(order)
train_data = train_data[:,order].copy()
train_labels = train_labels[order].copy()
is_noise = is_noise[order]

train_batches, test_batches = prepare_batches(train_data, train_labels, test_data, test_labels, batch_size)
train(net, num_epoch, train_batches, test_batches)

while True:
	N = int(np.floor(train_data.shape[1] / batch_size) * batch_size)
	cost = np.zeros(N)
	i = 0
	for batch in train_batches:
		net.train_batch(batch.data, batch.labels, TEST)
		cost[i:i+batch_size] = copy_to_cpu(net.layers[-1].cost).reshape(batch_size)
		i += batch_size
	order = np.argsort(cost)
	order = order[0:70000]
	train_data2 = train_data[:,order]
	train_labels2 = train_labels[order]
	print '# num of noisy labels:', np.sum(is_noise[order])
	N = int(np.floor(train_data2.shape[1] / batch_size) * batch_size)
	order = range(N)
	np.random.shuffle(order)
	train_data2 = train_data2[:,order].copy()
	train_labels2 = train_labels2[order].copy()

	train_batches2, test_batches2 = prepare_batches(train_data2, train_labels2, test_data, test_labels, batch_size)
	train(net, num_epoch, train_batches2, test_batches2)
