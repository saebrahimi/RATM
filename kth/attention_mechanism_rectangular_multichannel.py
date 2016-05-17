# coding: utf-8

# selective attention mechanism based on
# Gregor et al - "DRAW: A Recurrent Network For Image Generation"

import numpy as np

import theano
import theano.tensor as T


def batchdot(x, y):
    return (x.dimshuffle(0, 1, 2, 'x') *
            y.dimshuffle(0, 'x', 1, 2)).sum(
        axis=-2)


class SelectiveAttentionMechanism(object):

    def __init__(self, imsize, patchsize, nchannels=3, eps=1e-4):
        """Mechanism selecting glimpse from image based on attention input

        Args
        ----
            imsize: int or tuple specifying input image size
            patchsize: tuple, specifying number of rows and columns of patch
                to be selected
        """

        assert len(patchsize) == 2
        self.patchsize = patchsize
        self.eps = eps
        self.nchannels = nchannels

        if isinstance(imsize, int):
            self.imsize = (imsize, imsize)
        elif hasattr(imsize, '__iter__'):
            self.imsize = imsize
        else:
            raise ValueError(
                'imsize must either be int or iterable specifying Height, Width')

    def build_read_graph(self, images_var, attention_acts):
        fX, fY, muX, muY, gX, gY = self.compute_windows(
            attention_acts, with_mu=True)

        images_var.name = 'images_var'
        # reshape to BxCx0x1
        images = images_var.reshape((
            (images_var.shape[0], self.nchannels) + self.imsize))

        channel_patches = []
        # apply window filters to each channel separately
        for c in range(self.nchannels):
            channel_patches.append(
                batchdot(batchdot(fY, images[:, c]),
                         fX.dimshuffle(0, 2, 1)).dimshuffle(0, 'x', 1, 2))
        # and concatenate the patch channels
        patches = T.cast(T.concatenate(channel_patches, axis=1), 'float32')
        return (patches, muX, muY, gX, gY)

    def build_write_graph(self, patches_var, attention_acts):
        fX, fY = self.compute_windows(attention_acts)

        # reshape to BxCx0x1
        patches = patches_var.reshape((
            (patches_var.shape[0], self.nchannels) + self.patchsize))

        channel_images = []
        # apply window filters to each channel separately
        for c in range(self.nchannels):
            channel_images.append(
                batchdot(batchdot(fY.dimshuffle(0, 2, 1), patches[:, c]),
                         fX).dimshuffle(0, 'x', 1, 2))
        images = T.concatenate(channel_images, axis=1)
        return images

    def get_window_params_numpy(self, attention_acts):
        """Get attention params from 5-dim vector of activations

        Args
        ----
            attention_acts: numpy array of size Nx5 (columns corresponding
                to grid center horizontal, grid center vertical, sigma
                (isotropic variance), delta (stride), gamma (scalar intensity)
        """

        # get grid center coords
        gX = self.imsize[1] * attention_acts[:, 0]
        gY = self.imsize[0] * attention_acts[:, 1]

        sigmaX = np.absolute(attention_acts[:, 2]) / 2.
        deltaX = (self.imsize[1] - 1) / (self.patchsize[1] -
                                         1) * np.absolute(attention_acts[:, 3])
        gammas = np.absolute(attention_acts[:, 4])

        sigmaY = np.absolute(attention_acts[:, 5]) / 2.
        deltaY = (self.imsize[0] - 1) / (self.patchsize[0] -
                                         1) * np.absolute(attention_acts[:, 6])

        # I and J contain indices into the patch,
        # A and B indices into the input image
        I = np.arange(self.patchsize[1])[None, :, None]
        J = np.arange(self.patchsize[0])[None, :, None]

        # determine mean location
        muX = gX[:, None, None] + deltaX[:, None, None] * (
            I - self.patchsize[1] / 2. - .5)
        muY = gY[:, None, None] + deltaY[:, None, None] * (
            J - self.patchsize[0] / 2. - .5)

        return gX, gY, sigmaX, sigmaY, deltaX, deltaY, gammas, muX, muY

    def compute_windows(self, attention_acts, with_mu=False):
        """Get attention params from 5-dim vector of activations

        Args
        ----
            attention_acts: theano variable of size Nx5 (columns corresponding
                to grid center horizontal, grid center vertical, sigma
                (isotropic variance), delta (stride), gamma (scalar intensity)
        """

        # get grid center coords
        gX = self.imsize[1] * attention_acts[:, 0]
        gY = self.imsize[0] * attention_acts[:, 1]

        sigmaX = abs(attention_acts[:, 2]) / 2.
        deltaX = (self.imsize[1] - 1) / \
            (self.patchsize[1] - 1) * abs(attention_acts[:, 3])

        sigmaY = abs(attention_acts[:, 5]) / 2.
        deltaY = (self.imsize[0] - 1) / \
            (self.patchsize[0] - 1) * abs(attention_acts[:, 6])

        # I and J contain indices into the patch,
        # A and B indices into the input image
        I = T.arange(self.patchsize[1]).dimshuffle('x', 0, 'x')
        J = T.arange(self.patchsize[0]).dimshuffle('x', 0, 'x')
        A = T.arange(self.imsize[1]).dimshuffle('x', 'x', 0)
        B = T.arange(self.imsize[0]).dimshuffle('x', 'x', 0)

        # determine mean location
        muX = gX.dimshuffle(0, 'x', 'x') + deltaX.dimshuffle(
            0, 'x', 'x') * (I - self.patchsize[1] / 2. - .5)
        muY = gY.dimshuffle(0, 'x', 'x') + deltaY.dimshuffle(
            0, 'x', 'x') * (J - self.patchsize[0] / 2. - .5)

        # generate filter banks from the parameters and normalize them to sum 1
        fX = T.exp(-(T.addbroadcast(A, 1) - T.addbroadcast(muX, 2))**2 / (
            2 * sigmaX.dimshuffle(0, 'x', 'x')**2))
        fY = T.exp(-(T.addbroadcast(B, 1) - T.addbroadcast(muY, 2))**2 / (
            2 * sigmaY.dimshuffle(0, 'x', 'x')**2))
        fX /= fX.sum(axis=-1).dimshuffle(0, 1, 'x') + self.eps
        fY /= fY.sum(axis=-1).dimshuffle(0, 1, 'x') + self.eps
        if with_mu:
            return fX, fY, muX, muY, gX, gY
        else:
            return fX, fY

if __name__ == '__main__':

    theano.config.exception_verbosity = 'high'

    import matplotlib.pyplot as plt
    from scipy.misc import face
    plt.gray()

    numpy_rng = np.random.RandomState(1)

    # get example image
    im = face()[::4, ::4].astype(np.float32) / np.float32(255.)

    imsize = im.shape[:2]
    im = im.transpose(2, 0, 1)
    patchsize = (50, 50)
    nhid = 99
    batchsize = 10
    hids = theano.shared(numpy_rng.normal(
        loc=.5, scale=.5, size=(batchsize, nhid)).astype(
            theano.config.floatX),
        name='hids')

    L = theano.shared(
        (np.array([[.5, .5, 4.4,
                    # np.log(10.8/12.8),
                    1.8 / 12.8,
                    .1]],
                  dtype=np.float32) / (nhid * .5)).repeat(
            nhid, axis=0) + numpy_rng.uniform(
            low=-.05, high=.05, size=(nhid, 5)).astype(
            np.float32), name='L')

    attention_acts = T.dot(hids, L)

    images_var = T.fmatrix('images')

    fn = theano.function([], attention_acts)
    print 'fn(): {0}'.format(fn(), )

    attention_mechanism = SelectiveAttentionMechanism(
        imsize=imsize, patchsize=patchsize, nchannels=3)

    readfn = theano.function([images_var],
                             attention_mechanism.build_read_graph(images_var, attention_acts))

    patches_var = T.fmatrix('patches')
    patches, mux, muy, gX, gY = readfn(
        im.reshape(1, -1).repeat(batchsize, axis=0))
    print 'shape of patches: {0}'.format(patches.shape, )

    writefn = theano.function([patches_var],
                              attention_mechanism.build_write_graph(patches_var, attention_acts))
    write_imgs = writefn(patches.reshape(patches.shape[0], -1))
    print 'shape of write_imgs: {0}'.format(write_imgs.shape, )

    plt.clf()
    plt.subplot(4, batchsize, 1)
    plt.imshow(im.transpose(1, 2, 0), interpolation='nearest')
    plt.title('im')
    plt.axis('off')
    for i in range(batchsize):
        print i
        plt.subplot(4, batchsize, i + batchsize + 1)
        plt.imshow(patches[i].transpose(1, 2, 0), interpolation='nearest')
        plt.axis('off')
        plt.subplot(4, batchsize, i + 2 * batchsize + 1)
        plt.imshow(write_imgs[i].transpose(1, 2, 0), interpolation='nearest')
        plt.axis('off')
        plt.subplot(4, batchsize, i + 3 * batchsize + 1)
        rec_im = write_imgs[:i].sum(0)
        plt.imshow(rec_im.transpose(1, 2, 0), interpolation='nearest')
        plt.axis('off')

    plt.show(block=True)
    plt.savefig('attention_fixed_rectangular.png')
