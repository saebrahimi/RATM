import sys
sys.path.append('..')

import numpy as np
import theano
import theano.tensor as T

from common.model import Model


class RNN(Model):
    def __init__(self, nin, nout, nhid, numpy_rng, scale=1.0):
        self.nin = nin
        self.nout = nout
        self.nhid = nhid
        self.numpy_rng = numpy_rng
        self.scale = np.float32(scale)

        self.inputs = T.fmatrix('inputs')
        self.inputs.tag.test_value = numpy_rng.uniform(
            low=-1., high=1.,
            size=(16, 5 * self.nin)
        ).astype(np.float32)
        self.targets = T.fmatrix('targets')
        self.targets.tag.test_value = np.ones(
            (16, 5 * nout), dtype=np.float32)
        self.masks = T.bmatrix('masks')
        self.masks.tag.test_value = np.ones(
            (16, 5), dtype=np.int8)
        self.batchsize = self.inputs.shape[0]

        self.inputs_frames = self.inputs.reshape((
            self.batchsize, self.inputs.shape[1] / nin,
            nin)).dimshuffle(1, 0, 2)
        self.targets_frames = self.targets.reshape((
            self.batchsize, self.targets.shape[1] / nout,
            nout)).dimshuffle(1, 0, 2)
        self.masks_frames = self.masks.T

        self.h0 = theano.shared(value=np.ones(
            nhid, dtype=theano.config.floatX) * np.float32(.5), name='h0')
        self.win = theano.shared(value=self.numpy_rng.normal(
            loc=0, scale=0.001, size=(nin, nhid)
        ).astype(theano.config.floatX), name='win')
        self.wrnn = theano.shared(value=self.scale * np.eye(
            nhid, dtype=theano.config.floatX), name='wrnn')
        self.wout = theano.shared(value=self.numpy_rng.uniform(
            low=-0.01, high=0.01, size=(nhid, nout)
        ).astype(theano.config.floatX), name='wout')
        self.bout = theano.shared(value=np.zeros(
            nout, dtype=theano.config.floatX), name='bout')

        self.params = [self.win, self.wrnn, self.wout, self.bout]

        (self.hiddens, self.outputs), self.updates = theano.scan(
            fn=self.step, sequences=self.inputs_frames,
            outputs_info=[T.alloc(
                self.h0, self.batchsize, self.nhid), None])

        self._stepcosts = T.sum((self.targets_frames - self.outputs)**2, axis=2)
        self._cost = T.switch(self.masks_frames > 0, self._stepcosts, 0).mean()
        self._grads = T.grad(self._cost, self.params)

        self.getoutputs = theano.function(
            [self.inputs], self.outputs)

    def step(self, inp_t, h_tm1):
        pre_h_t = T.dot(inp_t, self.win) + T.dot(h_tm1, self.wrnn)
        h_t = T.switch(pre_h_t > 0, pre_h_t, 0)
        o_t = T.dot(h_t, self.wout) + self.bout
        return h_t, o_t
