from util import *

base_dir = '/scratch/sainaa/imagenet/'
#net = get_net(base_dir + 'checkpoint/reset-full-16')
#net = get_net(base_dir + 'checkpoint/reset-animal-16')
#net = get_net(base_dir + 'checkpoint/reset-dog-16')
net = get_net(base_dir + 'checkpoint/reset-device-16')

#train_dp, test_dp = get_dp(base_dir + 'train/')
#train_dp, test_dp = get_dp(base_dir + 'train-animal/')
#train_dp, test_dp = get_dp(base_dir + 'train-dog/')
train_dp, test_dp = get_dp(base_dir + 'train-device/')

#avail_labels = labels_from_datadir(base_dir + 'train-animal/')
#avail_labels = labels_from_datadir(base_dir + 'train-dog/')
#avail_labels = labels_from_datadir(base_dir + 'train-device/')
avail_labels = None

test_dp.reset()
total_case = 0
correct_case1 = 0
correct_case5 = 0
while True:
    correct1, correct5, numCase = test_error(net, test_dp, avail_labels)
    if test_dp.curr_epoch > 1:
        break
    total_case += numCase
    correct_case1 += correct1
    correct_case5 += correct5
    print 'done', test_dp.curr_epoch, test_dp.index, numCase
    print 'result:', (1.0 * correct_case1 / total_case), (1.0 * correct_case5 / total_case), total_case

