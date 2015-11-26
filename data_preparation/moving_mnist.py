# coding: utf-8
import cPickle as pickle
import gzip
import os

import numpy as np
import tables


class MovingMNIST(object):

    def __init__(self, mnist_path, numpy_rng):
        with gzip.open(mnist_path, 'rb') as f:
            sets = pickle.load(f)
        self.numpy_rng = numpy_rng

        self.partitions = {}
        for p, part in enumerate(('train', 'val', 'test')):
            self.partitions[part] = {}
            self.partitions[part]['images'] = sets[p][0].reshape(-1, 28, 28)
            self.partitions[part]['targets'] = sets[p][1]

    def get_batch(self, part, nvids, nframes, framesize, idx=None):
        vids = np.zeros((nvids, nframes, framesize, framesize))
        pos_init = self.numpy_rng.randint(framesize - 28, size=(nvids, 2))
        pos = np.zeros((nvids, nframes, 2), dtype=np.int32)
        pos[:,0] = pos_init

        posmax = framesize - 29

        d = self.numpy_rng.randint(low=-15, high=15, size=(nvids, 2))
        if idx is None:
            idx = self.numpy_rng.randint(
                self.partitions[part]['images'].shape[0],
                size=nvids)
        for t in range(vids.shape[1]):
            dtm1 = d
            d = self.numpy_rng.randint(low=-15, high=15, size=(nvids, 2))
            for i in range(nvids):
                vids[i,t,
                    pos[i,t,0]:pos[i,t,0]+28,
                    pos[i,t,1]:pos[i,t,1]+28] = \
                    self.partitions[part]['images'][idx[i]]
            if t < nframes-1:
                pos[:,t+1] = pos[:,t] + .1 * d + .9 * dtm1

                # check for proper position (reflect if necessary)
                reflectidx = np.where(pos[:,t+1] > posmax)
                pos[:,t+1][reflectidx] = (posmax - (pos[:,t+1][reflectidx] % posmax))

                reflectidx = np.where(pos[:,t+1] < 0)
                pos[:,t+1][reflectidx] = - pos[:,t+1][reflectidx]

        # return videos, ground truth positions and targets
        return vids, pos, self.partitions[part]['targets'][idx]

    def dump_test_set(self, h5filepath, nframes, framesize):
        # set rng to a hardcoded state, so we always have the same test set!
        self.numpy_rng.seed(1)
        with tables.openFile(h5filepath, 'w') as h5file:

            h5file.createArray(h5file.root, 'test_targets',
                               self.partitions['test']['targets'])

            vids = h5file.createCArray(
                h5file.root,
                'test_images',
                tables.Float32Atom(),
                shape=(10000,
                       nframes, framesize, framesize),
                filters=tables.Filters(complevel=5, complib='zlib'))

            pos = h5file.createCArray(
                h5file.root,
                'test_pos',
                tables.UInt16Atom(),
                shape=(10000,
                       nframes, 2),
                filters=tables.Filters(complevel=5, complib='zlib'))
            for i in range(100):
                print i
                (vids[i*100:(i+1)*100],
                 pos[i*100:(i+1)*100], _) = self.get_batch(
                     'test', 100, nframes, framesize,
                     idx=np.arange(i*100,(i+1)*100))
                h5file.flush()





if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # TODO: point to mnist.pkl.gz
    # available here: http://deeplearning.net/data/mnist/mnist.pkl.gz
    dataset_path = '/path/to/mnist.pkl.gz'

    test_set_path = '/path/to/moving_mnist_test_set_nframes{0}_framesize{1}.h5'.format(
        30, 100)

    os.system('mkdir -p tmp')
    numpy_rng = np.random.RandomState(1)

    dataset = MovingMNIST(
        mnist_path=dataset_path,
        numpy_rng=numpy_rng)

    print "dumping test set..."
    dataset.dump_test_set(test_set_path, nframes=30, framesize=100)
    print "done (with dumping test set)"


    print "generating train batch..."
    vids, pos, targets = dataset.get_batch(
        'train', 16, 30, 100)
    print "done (with generating train batch)"

    plt.gray()
    plt.show()
    for t in range(vids.shape[1]):
        for i in range(vids.shape[0]):
            plt.subplot(5,5,i+1)
            plt.imshow(vids[i,t])
            plt.title(targets[i])
            ax = plt.gca()
            ax.add_patch(
                Rectangle(
                    pos[i,t][::-1],
                    28, 28, facecolor='r', edgecolor='r'))
            plt.draw()
            plt.axis('off')
        plt.savefig('tmp/frame{0:03d}.png'.format(t))
    os.system('convert -loop 0 -delay 30 tmp/frame*.png tmp/vid.gif')

