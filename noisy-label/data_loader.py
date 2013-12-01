import numpy as np
import scipy.io
import pycuda.driver as cuda
from pycuda import gpuarray
from fastnet import util

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

def prepare_batches(data, labels, batch_size):
    data = data.copy()
    labels = labels.copy()
    N = data.shape[1]
    batches = list()
    ind = 0
    while ind + batch_size <= N:
        batch = BatchData(data[:,ind:ind+batch_size], \
							labels[ind:ind+batch_size])
        batches.append(batch)
        ind += batch_size

    return batches

def load_cifar10():
    base_dir = '/home/sainbar/data/cifar-10/train/'
    batch_meta = util.load(base_dir + 'batches.meta')
    data_file1 = util.load(base_dir + 'data_batch_1')
    data_file2 = util.load(base_dir + 'data_batch_2')
    data_file3 = util.load(base_dir + 'data_batch_3')
    data_file4 = util.load(base_dir + 'data_batch_4')
    data_file5 = util.load(base_dir + 'data_batch_5')
    data_file6 = util.load(base_dir + 'data_batch_6')
    labels1 = np.array(data_file1['labels'])
    labels2 = np.array(data_file2['labels'])
    labels3 = np.array(data_file3['labels'])
    labels4 = np.array(data_file4['labels'])
    labels5 = np.array(data_file5['labels'])
    labels6 = np.array(data_file6['labels'])
    
    train_data = np.concatenate((data_file1['data'],data_file2['data'],data_file3['data'],data_file4['data'],data_file5['data']), axis=1)
    train_labels = np.concatenate((labels1,labels2,labels3,labels4,labels5), axis=1)
    test_data = data_file6['data']
    test_labels = labels6
    train_data = train_data.astype(np.float32).copy()
    test_data = test_data.astype(np.float32).copy()
    return train_data, train_labels, test_data, test_labels

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
    
def load_noisy_labeled():
    data_file = scipy.io.loadmat('/home/sainbar/data/cifar-10/geoff-neg-150k.mat')
    data = data_file['data']
    labels = data_file['labels']
    data = data.astype(np.float32)
    labels = labels.astype(np.float32) - 1 # important to substract 1
    data = data.copy()
    labels = labels.copy()
    labels = labels.reshape(labels.shape[1])

    return data, labels

