import fastnet.net
import numpy as np
import data_loader
from fastnet.layer import TRAIN, TEST

def get_scores(net, batches):
	batch_size = batches[0].data.shape[1]
	N = len(batches) * batch_size
	score = np.zeros(N)
	i = 0
	for batch in batches:
		net.train_batch(batch.data, batch.labels, TEST)
		score[i:i+batch_size] = -1 * data_loader.copy_to_cpu(net.layers[-1].cost).reshape(batch_size)
		i += batch_size
	return score

def get_easy_data(data, labels, score, M):
	order = np.argsort(-score)
	order = order[0:M]
	data2 = data[:,order]
	labels2 = labels[order]
	return data2, labels2

def get_easy_data_balanced(data, labels, score, M):
	assert M % 10 == 0
	data2 = np.zeros((3072, M/10, 10)).astype(np.float32)
	labels2 = np.zeros((M/10, 10)).astype(np.float32)
	for k in range(10):
		d = data[:,labels == k]
		l = labels[labels == k]
		s = score[labels == k]
		d2, l2 = get_easy_data(d,l,s,M/10)
		data2[:,:,k] = d2
		labels2[:,k] = l2
	data2 = data2.reshape(3072, M)
	labels2 = labels2.reshape(M)
	return data2, labels2
	
def get_new_batches(net, batch_size, train_batches, train_data, train_labels, M):
	scores = get_scores(net, train_batches)
	#scores[0:20000] = 10
	train_data2, train_labels2 = get_easy_data_balanced(train_data, train_labels, scores, M)
	order = range(M)
	np.random.shuffle(order)
	train_data2 = train_data2[:,order]
	train_labels2 = train_labels2[order]
	train_batches2 = data_loader.prepare_batches(train_data2, train_labels2, batch_size)
	print '# train:', train_data2.shape[1], 'samples', len(train_batches2), 'batches'
	return train_batches2
	