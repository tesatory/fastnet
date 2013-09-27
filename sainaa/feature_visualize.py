import fastnet.net
import fastnet.layer
from util import *

def reverse_fprop(l, input, output):
    print 'reversing', l.name
    if l.type == 'conv':
        #gpu_copy_to(output, l.tmp)
        #add_vec_to_rows(l.tmp, -l.bias)
        #gpu_copy_to(l.tmp, output)
        l.bprop(output, input, output, input)
    elif l.type == 'pool':
        l.bprop(output, input, output, input)
    elif l.type == 'neuron':
        l.bprop(output, input, output, input)
    elif l.type == 'cmrnorm':
        l.bprop(output, input, output, input)
    else:
        pass

def reverse_fprop_net(net, data, start_from):
    start = False
    a = range(1, len(net.layers))
    a.reverse()
    for i in a:
        l = net.layers[i]
        if l.name == start_from:
            start = True
        if start:
            input = net.outputs[i-1]
            output = net.outputs[i]
            reverse_fprop(l, input, output)

net = get_net('/scratch/sainaa/imagenet/checkpoint/reset-dog-16')
train_dp, test_dp = get_dp('/scratch/sainaa/imagenet/train-dog/')
batch = train_dp.get_next_batch(128)
data = batch.data
labels = batch.labels
net.train_batch(data, labels, TEST)
reverse_fprop_net(net, data, 'conv2')
plot_images(data)
