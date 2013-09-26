import zipfile
import cPickle
import numpy
import glob
from os.path import basename
import pycuda.driver as cuda
import fastnet.net
import fastnet.data
from fastnet.layer import TRAIN, TEST

def to_cpu(gpu_data):
    data = numpy.zeros(gpu_data.shape).astype(numpy.float32)
    cuda.memcpy_dtoh(data, gpu_data.gpudata)
    return data

def open_checkpoint(chkpnt_path):
    zf = zipfile.ZipFile(chkpnt_path, 'r')
    layers = cPickle.load(zf.open('layers'))
    return layers

def get_net(path):
    layers = open_checkpoint(path)
    model = dict()
    model['layers'] = layers
    image_sz = 224
    batch_sz = 128
    return fastnet.net.FastNet(0.1, (3, image_sz, image_sz, batch_sz), model)

def get_dp(data_dir = '/scratch/sainaa/imagenet/train/'):
    train_range = range(101, 1301) #1,2,3,....,40
    test_range = range(1, 101) #41, 42, ..., 48
    data_provider = 'imagenet'
    train_dp = fastnet.data.get_by_name(data_provider)(data_dir,train_range)
    test_dp = fastnet.data.get_by_name(data_provider)(data_dir, test_range)
    return train_dp, test_dp

def labels_from_datadir(data_dir):
    train_dp, test_dp = get_dp(data_dir)
    dirs = glob.glob(data_dir + '/n*')
    labels = list()
    for synid in dirs:
        synid = basename(synid)[1:]
        lbl = train_dp.dp.batch_meta['synid_to_label'][synid]
        labels.append(lbl)
    print labels

def test_error(net, dp, output_restrict = None, batch = None):
    if batch == None:
        batch = dp.get_next_batch(128)
    net.train_batch(batch.data, batch.labels, TEST)
    
    output = to_cpu(net.output)
    labels = to_cpu(batch.labels)
    
    correct1 = 0
    correct5 = 0
    
    if output_restrict != None:
        output[output_restrict,:] = output[output_restrict,:] + 10
    
    max_labels = numpy.argsort(output, 0)
    for i in range(output.shape[1]):
        if max_labels[999,i] == labels[i]:
            correct1 += 1
            correct5 += 1
        elif max_labels[998,i] == labels[i]:
            correct5 += 1
        elif max_labels[997,i] == labels[i]:
            correct5 += 1
        elif max_labels[996,i] == labels[i]:
            correct5 += 1
        elif max_labels[995,i] == labels[i]:
            correct5 += 1
    return correct1, correct5, output.shape[1]

