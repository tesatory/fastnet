import scipy.io
import data_loader

def save_net(net, path):
	d = dict()
	d['w1'] = data_loader.copy_to_cpu(net.layers[1].weight)
	d['w2'] = data_loader.copy_to_cpu(net.layers[5].weight)
	d['w3'] = data_loader.copy_to_cpu(net.layers[9].weight)
	d['w4'] = data_loader.copy_to_cpu(net.layers[12].weight)
	d['b1'] = data_loader.copy_to_cpu(net.layers[1].bias)
	d['b2'] = data_loader.copy_to_cpu(net.layers[5].bias)
	d['b3'] = data_loader.copy_to_cpu(net.layers[9].bias)
	d['b4'] = data_loader.copy_to_cpu(net.layers[12].bias)
	scipy.io.savemat(path, d)

def load_net(net, path):
	d = scipy.io.loadmat(path)
	net.layers[1].weight = data_loader.copy_to_gpu(d['w1'])
	net.layers[5].weight = data_loader.copy_to_gpu(d['w2'])
	net.layers[9].weight = data_loader.copy_to_gpu(d['w3'])
	net.layers[12].weight = data_loader.copy_to_gpu(d['w4'])
	net.layers[1].bias = data_loader.copy_to_gpu(d['b1'])
	net.layers[5].bias = data_loader.copy_to_gpu(d['b2'])
	net.layers[9].bias = data_loader.copy_to_gpu(d['b3'])
	net.layers[12].bias = data_loader.copy_to_gpu(d['b4'])
