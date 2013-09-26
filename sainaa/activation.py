import zipfile
import cPickle
import pycuda.driver as cuda

def build_image(array):
  array = array - array.min()
  array = array / array.max()
  if len(array.shape) == 4:
    filter_size = array.shape[1]
  else:
    filter_size = array.shape[0]
  
  num_filters = array.shape[-1]
  num_cols = util.divup(2000, filter_size)
  num_rows = util.divup(num_filters, num_cols)

  if len(array.shape) == 4:
    big_pic = np.zeros((3, (filter_size + 1) * num_rows, (filter_size + 1) * num_cols))
  else:
    big_pic = np.zeros((filter_size * num_rows, filter_size * num_cols))
  
  for i in range(num_rows):
    for j in range(num_cols):
      idx = i * num_cols + j
      if idx >= num_filters: break
      x = i*(filter_size + 1)
      y = j*(filter_size + 1)
      if len(array.shape) == 4:
        big_pic[:, x:x+filter_size, y:y+filter_size] = array[:, :, :, idx]
      else:
        big_pic[x:x+filter_size, y:y+filter_size] = array[:, :, idx]
  
  if len(array.shape) == 4:
    return big_pic.transpose(1, 2, 0)
  return big_pic

def plot_images(feature):
    a = out_big[feature,:]
    b = argsort(a)
    c = b[-80:]
    print labels_big.flatten()[(c)]
    d = data_big[:,(c)].reshape(3,224,224,-1)
    img = build_image(d)
    imshow(img)
    

net = get_net("/scratch/sainaa/checkpoint/simple_7+12")
train_dp = get_dp()

N = 500
out_big1 = numpy.zeros((96,batch_sz*N))
out_big2 = numpy.zeros((256,batch_sz*N))
out_big3 = numpy.zeros((384,batch_sz*N))
out_big4 = numpy.zeros((384,batch_sz*N))
out_big5 = numpy.zeros((256,batch_sz*N))
labels_big = numpy.zeros((batch_sz*N,1))
#data_big = numpy.zeros((3*224*224,128*N))

for n in range(N):
    print n    
    train_data = train_dp.get_next_batch(batch_sz)
    while train_data.labels.shape[0] < batch_sz:
        print train_data.labels.shape[0]
        train_data = train_dp.get_next_batch(batch_sz)

    data, labels = net.prepare_for_train(train_data.data, train_data.labels)
    net.fprop(data,net.output)
    
    out = net.outputs[2]  # layer 1 output
    print out.shape
    out = out.reshape(96,55*55,batch_sz)
    out2 = numpy.zeros(out.shape).astype(numpy.float32)
    cuda.memcpy_dtoh(out2,out.gpudata)
    out = out2.mean(1)
    out_big1[:,n*batch_sz:(n+1)*batch_sz] = out
    
    out = net.outputs[6]  # layer 2 output
    print out.shape
    out = out.reshape(256,27*27,batch_sz)
    out2 = numpy.zeros(out.shape).astype(numpy.float32)
    cuda.memcpy_dtoh(out2,out.gpudata)
    out = out2.mean(1)
    out_big2[:,n*batch_sz:(n+1)*batch_sz] = out
    
    out = net.outputs[10]  # layer 3 output
    print out.shape
    out = out.reshape(384,13*13,batch_sz)
    out2 = numpy.zeros(out.shape).astype(numpy.float32)
    cuda.memcpy_dtoh(out2,out.gpudata)
    out = out2.mean(1)
    out_big3[:,n*batch_sz:(n+1)*batch_sz] = out
    
    out = net.outputs[12]  # layer 4 output
    print out.shape
    out = out.reshape(384,13*13,batch_sz)
    out2 = numpy.zeros(out.shape).astype(numpy.float32)
    cuda.memcpy_dtoh(out2,out.gpudata)
    out = out2.mean(1)
    out_big4[:,n*batch_sz:(n+1)*batch_sz] = out
    
    out = net.outputs[14]  # layer 5 output
    print out.shape
    out = out.reshape(256,13*13,batch_sz)
    out2 = numpy.zeros(out.shape).astype(numpy.float32)
    cuda.memcpy_dtoh(out2,out.gpudata)
    out = out2.mean(1)
    out_big5[:,n*batch_sz:(n+1)*batch_sz] = out

    #data2 = numpy.zeros(data.shape).astype(numpy.float32)
    #cuda.memcpy_dtoh(data2,data.gpudata)
    #data_big[:,n*batch_sz:(n+1)*batch_sz] = data2
    
    labels2 = numpy.zeros(labels.shape).astype(numpy.float32)
    cuda.memcpy_dtoh(labels2,labels.gpudata)
    labels_big[n*batch_sz:(n+1)*batch_sz,:] = labels2

