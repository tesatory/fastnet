from util import *

base_dir = '/scratch/sainaa/cifar-10/'
net = get_net(base_dir + 'checkpoint/cifar-test-1-11')
train_dp, test_dp = get_dp_cifar(base_dir + 'train/')

test_dp.reset()
total_case = 0
correct_case = 0
while True:
    batch = test_dp.get_next_batch(128)
    net.train_batch(batch.data, batch.labels, TEST)
    x, correct, numCase = net.get_batch_information()

    if test_dp.curr_epoch > 1:
        break
    total_case += numCase
    correct_case += correct * numCase
    print 'result:', (1.0 * correct_case / total_case), total_case

