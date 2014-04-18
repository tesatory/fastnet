import sys
import numpy
from fastnet.cuda_kernel import *
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import fastnet.net
import fastnet.layer
import pylab
import matplotlib.pyplot as plt
from util import *

def reverse_fprop(l, feature, feature_out, input, output):
    print 'reversing', l.name
    if l.type == 'conv':
        if False:
            feature_num = l.outputShape[0]
            batch_size = l.outputShape[3]
            w = to_cpu(l.weight)
            w2 = (w**2).sum(0).reshape(feature_num,1)
            f = to_cpu(feature)
            f = f.reshape(feature_num,-1)
            f = numpy.divide(f,w2)
            f = f.reshape(-1,batch_size)
            feature = to_gpu(f)
        if False and l.name != 'conv1':
            gpu_copy_to(feature, l.tmp)
            add_vec_to_rows(l.tmp, -l.bias)
            gpu_copy_to(l.tmp, feature)
        l.bprop(feature, input, output, feature_out)
    elif l.type == 'pool':
        l.bprop(feature, input, output, feature_out)
    elif l.type == 'neuron':
        #gpu_copy_to(feature, feature_out)
        l.bprop(feature, input, output, feature_out)
        gpu_copy_to(feature_out, feature)
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
            if True:
                feature_num = l.outputShape[0]
                feature_map_size = l.outputShape[1]
                batch_size = l.outputShape[3]
                f = f.reshape(feature_num, feature_map_size**2, batch_size)
                f2 = numpy.zeros(f.shape)
                f2.fill(0)
                max_act = numpy.zeros(128)
                for img_id in range(128):
                    max_act[img_id] = f[feature_id,:,img_id].max()
                    max_pos = f[feature_id,:,img_id].argmax()
                    f2[feature_id, max_pos, img_id] = 1
                f2 = f2.reshape(feature_num*feature_map_size**2, batch_size)
                feature = to_gpu(f2)
            else:
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
    return feature_out, max_act

layer_name = sys.argv[1]
net = get_net(sys.argv[2])
#for l in net.layers:
#    if l.type in ('cmrnorm','rnorm'):
#        l.pow = 0


#data = to_gpu(numpy.load('./batch_data.npy'))
#labels = to_gpu(numpy.load('./batch_labels.npy'))
train_dp, test_dp = get_dp('/ssd/fergusgroup/sainaa/imagenet/train/')
batch_num = 10
fnum = 40
ff = numpy.zeros((3*224*224,128,batch_num,fnum))
aa = numpy.zeros((128,batch_num,fnum))
for batch_ind in range(batch_num):
    batch = test_dp.get_next_batch(128)
    while batch.data.shape[1] != 128:
        print 'miss'
        batch = test_dp.get_next_batch(128)
    net.train_batch(batch.data, batch.labels, TEST)
    #net.train_batch(data, labels, TEST)
    for feature_id in range(fnum):
        f,max_act = reverse_fprop_net(net, layer_name, feature_id)
        ff[:,:,batch_ind,feature_id] = to_cpu(f)
        aa[:,batch_ind,feature_id] = max_act

ff = ff.reshape(-1,128*batch_num,fnum)
aa = aa.reshape(128*batch_num,fnum)
g = numpy.zeros((3*224*224,fnum,10))
for feature_id in range(fnum):
    a = aa[:,feature_id].argsort()
    g[:,feature_id,:] = ff[:,a[-10:],feature_id]
g = g.reshape(-1,10*fnum)
g = g - g.min(0,keepdims=True)
g = g / g.max(0,keepdims=True)
plot_images(g)
if len(sys.argv) > 3:
    pylab.savefig(sys.argv[3] + '-' + layer_name, dpi=600, bbox_inches=0)


