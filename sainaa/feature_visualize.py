import numpy
from fastnet.cuda_kernel import *
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import fastnet.net
import fastnet.layer
from util import *

def reverse_fprop(l, feature, feature_out, input, output):
    print 'reversing', l.name
    if l.type == 'conv':
        #gpu_copy_to(feature, l.tmp)
        #add_vec_to_rows(l.tmp, -l.bias)
        #gpu_copy_to(l.tmp, feature)
        l.bprop(feature, input, output, feature_out)
    elif l.type == 'pool':
        l.bprop(feature, input, output, feature_out)
    elif l.type == 'neuron':
        l.bprop(feature, input, output, feature_out)
        #l.fprop(feature, feature_out)
    elif l.type == 'cmrnorm':
        gpu_copy_to(feature, feature_out)
    else:
        pass

def reverse_fprop_net(net, start_from, feature_id):
    start = False
    max_img = -1
    net.features = list()
    a = range(1, len(net.layers))
    a.reverse()
    for i in a:
        l = net.layers[i]
        if l.name == start_from:
            start = True
            f = to_cpu(net.outputs[i])
            #feature_num = l.outputShape[0]
            #feature_map_size = l.outputShape[1]
            #batch_size = l.outputShape[3]
            #f = f.reshape(feature_num, feature_map_size**2, batch_size)
            #max_act = f[feature_id,:,:].max()
            #max_img = f[feature_id,:,:].max(0).argmax()
            #max_pos = f[feature_id,:,max_img].argmax()
            #assert f[feature_id, max_pos, max_img] == max_act
            #f.fill(0)
            #f[feature_id, max_pos, max_img] = max_act
            #f = f.reshape(feature_num*feature_map_size**2, batch_size)
            feature = to_gpu(f)
        if start:
            net.features.append(feature)
            input = net.outputs[i-1]
            output = net.outputs[i]
            feature_out = gpuarray.empty(input.shape, numpy.float32)
            reverse_fprop(l, feature, feature_out, input, output)
            feature = feature_out

    net.features.append(feature_out)
    net.features.reverse()
    return feature_out, max_img

net = get_net('/scratch/sainaa/imagenet/checkpoint/reset-dog-16')
data = to_gpu(numpy.load('./batch_data.npy'))
labels = to_gpu(numpy.load('./batch_labels.npy'))
net.train_batch(data, labels, TEST)
f,img_id = reverse_fprop_net(net, 'conv5', 0)
plot_images(f)

