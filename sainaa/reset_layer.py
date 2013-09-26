import zipfile
import cPickle
import numpy as np
from util import *

def randn(shape, dtype):
    np.random.seed(0)
    return np.require(np.random.randn(*shape), dtype=dtype, requirements='C')

chkpnt_path = '/scratch/sainaa/imagenet/checkpoint/normal-19ep'
layers = open_checkpoint(chkpnt_path)
initW = 0.01

assert len(layers) == 22
for li in range(13,22):
    l = layers[li]
    if 'weight' in l.keys():
        print "resetting", l['name']
        l['weight'] = randn(l['weight'].shape, np.float32) * initW
        l['bias'] = np.zeros(l['bias'].shape, np.float32)

with zipfile.ZipFile(chkpnt_path + '-5reset', mode='w') as output:
    output.writestr('layers', cPickle.dumps(layers, protocol=-1))
    for k in zf.namelist():
        if k != 'layers':
            output.writestr(k, zf.open(k).read())
