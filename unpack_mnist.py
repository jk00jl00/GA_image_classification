import cPickle, gzip, numpy
from imageio import imwrite

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

# convert both train and test to png as images
x = numpy.concatenate((train_set[0]*255,valid_set[0]*255,test_set[0][:3000,:]*255))
for i in range(20):
  imwrite('mnist_batch_'+`i`+'.png', x[3000*i:3000*(i+1),:])
imwrite('mnist_batch_'+`20`+'.png', x[60000:,:]) # test set

# dump the labels
L = 'labels=' + `list(numpy.concatenate((train_set[1],valid_set[1],test_set[1])))` + ';\n'
open('mnist_labels.py', 'w').write(L)

