import sys
import zipfile
import cPickle
import numpy as np
import matplotlib.pyplot as plt
import pylab
import pandas

def _load_series(data, scale=1):
    lp = [t['logprob'] for t,count,elapsed in data]
    counts = np.array([count for t,count,elapsed in data]).cumsum()
    examples = counts * scale
    
    elapsed = np.array([elapsed for t,count,elapsed in data])
    logprob = np.array([t[0] for t in lp])
    prec = np.array([t[1] for t in lp])
    return pandas.DataFrame({'lp' : logprob, 'pr' : prec, 'elapsed' : elapsed, 'examples' : examples})

pylab.ion()

chkpnt_path = sys.argv[1]
zf = zipfile.ZipFile(chkpnt_path, 'r')
train_outputs = cPickle.load(zf.open('train_outputs'))
test_outputs = cPickle.load(zf.open('test_outputs'))
train_df = _load_series(train_outputs)
test_df = _load_series(test_outputs)

batch_per_epoch = 1.0 * 1e+6
train_error = pandas.Series(train_df['pr'].values, train_df['examples'].values / batch_per_epoch)
test_error = pandas.Series(test_df['pr'].values, test_df['examples'].values * 100 / batch_per_epoch)
pandas.rolling_mean(train_error, 10000).plot()
pandas.rolling_mean(test_error, 100).plot()

if len(sys.argv) > 2:
    pylab.savefig(sys.argv[2], bbox_inches=0)