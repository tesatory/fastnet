import fastnet.net
from fastnet.layer import TRAIN, TEST

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
