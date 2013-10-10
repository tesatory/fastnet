import numpy
from fastnet.cuda_kernel import *
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import fastnet.net
import fastnet.layer
from util import *

def reverse_conv_batch(l, feature, feature_out): 
    img_size = 32
    color_num = 3
    feature_num = 32
    feature_size = 32
    padding = 2
    batch_size = feature.shape[1]

    feature = to_cpu(feature).reshape(32, 32, 32, batch_size)
    weight = to_cpu(l.weight)
    weight = weight.reshape(3, 5, 5, 32)
    out = numpy.zeros((3, 32, 32, batch_size))
    for fi in range(32):
        print fi
        for y in range(32):
            for x in range(32):
                for wy in range(5):
                    for wx in range(5):
                        for c in range(3):
                            if wy+y-padding >=0 and wy+y-padding < 32 and wx+x-padding >=0 and wx+x-padding < 32:
                                out[c, wy+y-padding, wx+x-padding,:] += weight[c, wy, wx, fi] * feature[fi, y, x,:]

    out = out.reshape(3*32*32,batch_size)
    out = to_gpu(out)
    gpu_copy_to(out, feature_out)

def reverse_fprop(l, feature, feature_out, input, output):
    print 'reversing', l.name
    if l.type == 'conv':
        #gpu_copy_to(feature, l.tmp)
        #add_vec_to_rows(l.tmp, -l.bias)
        #gpu_copy_to(l.tmp, feature)
        l.bprop(feature, input, output, feature_out)
        #reverse_conv_batch(l, feature, feature_out)
    elif l.type == 'pool':
        l.bprop(feature, input, output, feature_out)
    elif l.type == 'neuron':
        l.fprop(feature, feature_out)
    elif l.type in ('cmrnorm', 'rnorm'):
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
            # find max activation
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
            #f[feature_id, max_pos, max_img] = 1
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

def plot_layer(net, layer_name, k = 10):
    g = numpy.zeros((data.shape[0],k))
    for i in range(k):
        f,img_id = reverse_fprop_net(net, layer_name, i)
        f = to_cpu(f)
        g[:,i] = f[:,img_id]
    plot_images(g)
    return g

net = get_net('/scratch/sainaa/imagenet/checkpoint/long-train-1')
#net = get_net('/scratch/sainaa/cifar-10/checkpoint/cifar-test-1-61')
data = to_gpu(numpy.load('./batch_data.npy'))
labels = to_gpu(numpy.load('./batch_labels.npy'))
#data = to_gpu(numpy.load('./batch_data_cifar.npy'))
#labels = to_gpu(numpy.load('./batch_labels_cifar.npy'))
#for l in net.layers:
#    if l.type in ('cmrnorm','rnorm'):
#        l.pow = 0
net.train_batch(data, labels, TEST)
g = plot_layer(net,'conv2', 1)
