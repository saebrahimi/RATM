#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict
import sys
sys.path.append('../..')
import numpy as np
import theano
import theano.tensor as T

from common.layers import (AffineLayer, Clip, ConvBiasLayer, ConvLayer,
                           MaxPoolLayer, Relu, Softmax, Reshape)
from common.model import Model


class HumanConvNet(Model):

    def __init__(self, name, nout, numpy_rng, theano_rng, batchsize=128):
        # CALL PARENT CONSTRUCTOR TO SETUP CONVENIENCE FUNCTIONS
        # (SAVE/LOAD, ...)
        super(HumanConvNet, self).__init__(name=name)

        self.numpy_rng = numpy_rng
        self.batchsize = batchsize
        self.theano_rng = theano_rng
        self.mode = theano.shared(np.int8(0), name='mode')
        self.nout = nout

        self.inputs = T.ftensor4('inputs')
        self.inputs.tag.test_value = numpy_rng.randn(
            self.batchsize, 1, 28, 28).astype(np.float32)

        self.targets = T.ivector('targets')
        self.targets.tag.test_value = numpy_rng.randint(
            nout, size=self.batchsize).astype(np.int32)

        self.layers = OrderedDict()

        self.layers['conv0'] = ConvLayer(
            rng=self.numpy_rng,
            inputs=self.inputs,
            filter_shape=(128, 1, 5, 5),
            image_shape=(None, 1, 28, 28),
            name='conv0',
            pad=2
        )

        self.layers['maxpool0'] = MaxPoolLayer(
            inputs=self.layers['conv0'],
            pool_size=(2, 2),
            stride=(2, 2),
            name='maxpool0'
        )

        self.layers['bias0'] = ConvBiasLayer(
            inputs=self.layers['maxpool0'],
            name='bias0'
        )

        self.layers['relu0'] = Relu(
            inputs=self.layers['bias0'],
            name='relu0'
        )

        self.layers['conv1'] = ConvLayer(
            rng=self.numpy_rng,
            inputs=self.layers['relu0'],
            filter_shape=(64, 128, 3, 3),
            name='conv1',
            pad=1
        )

        self.layers['maxpool1'] = MaxPoolLayer(
            inputs=self.layers['conv1'],
            pool_size=(2, 2),
            stride=(2, 2),
            name='maxpool1'
        )

        self.layers['bias1'] = ConvBiasLayer(
            inputs=self.layers['maxpool1'],
            name='bias1'
        )

        self.layers['relu1'] = Relu(
            inputs=self.layers['bias1'],
            name='relu1'
        )

        self.layers['reshape1'] = Reshape(
            inputs=self.layers['relu1'],
            shape=(self.layers['relu1'].outputs_shape[0],
                   np.prod(self.layers['relu1'].outputs_shape[1:])),
            name='reshape1'
        )

        self.layers['fc2'] = AffineLayer(
            rng=self.numpy_rng,
            inputs=self.layers['reshape1'],
            nouts=256,
            name='fc2'
        )

        self.layers['relu2'] = Relu(
            inputs=self.layers['fc2'],
            name='relu2'
        )

        self.layers['fc3'] = AffineLayer(
            rng=self.numpy_rng,
            inputs=self.layers['relu2'],
            nouts=self.nout,
            name='fc3'
        )

        self.layers['softmax3'] = Softmax(
            inputs=self.layers['fc3'],
            name='softmax3'
        )

        self.layers['clip3'] = Clip(
            inputs=self.layers['softmax3'],
            name='clip3',
            min_val=1e-6, max_val=1-1e-6)

        self.probabilities = self.forward(self.inputs)

        self._cost = T.nnet.categorical_crossentropy(
            self.probabilities, self.targets).mean()

        self.classification = T.argmax(self.probabilities, axis=1)

        self.params = []
        for l in self.layers.values():
            self.params.extend(l.params)

        self._grads = T.grad(self._cost, self.params)

        self.classify = theano.function(
            [self.inputs], self.classification,
            )

    def forward(self, inputs):
        for l, layer in enumerate(self.layers.values()):
            print l, layer
            if l > 0:
                layer_outp = layer.forward(layer_outp)
            else:
                layer_outp = layer.forward(inputs)
        return layer_outp

    def forward_from_to(self, inputs, from_layer_name=None, to_layer_name=None):
        if from_layer_name is None:
            from_layer_name = self.layers.keys()[0]
        if to_layer_name is None:
            to_layer_name = self.layers.keys()[-1]
        l_from = self.layers.keys().index(from_layer_name)
        l_to = self.layers.keys().index(to_layer_name)
        for l, layer in zip(range(l_from, l_to+1), self.layers.values()[l_from:l_to+1]):
            print 'forward_from_to():', l, layer
            if l > l_from:
                layer_outp = layer.forward(layer_outp)
            else:
                layer_outp = layer.forward(inputs)
        return layer_outp

    def compute_01_loss(self, inputs, targets):
        # switch to test mode
        self.mode.set_value(np.int8(1))
        inputs = inputs.reshape((inputs.shape[0], ) +
                                self.layers.values()[0].inputs_shape[1:])
        predictions = np.zeros((inputs.shape[0],), dtype=np.int32)
        for b in range(inputs.shape[0] // self.batchsize):
            predictions[
                b * self.batchsize:
                (b + 1) * self.batchsize] = self.classify(
                inputs[b * self.batchsize: (b + 1) * self.batchsize])
        if inputs.shape[0] % self.batchsize:
            last_batch = inputs[-(inputs.shape[0] % self.batchsize):]
            predictions[-(inputs.shape[0] % self.batchsize):] = self.classify(
                last_batch)
        # switch to train mode
        self.mode.set_value(np.int8(0))

        return np.sum(targets != predictions) / np.float32(inputs.shape[0])
