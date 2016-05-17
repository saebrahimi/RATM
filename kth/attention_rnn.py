#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append('..')

import numpy as np
import theano
from theano.printing import Print
import theano.tensor as T
theano.config.compute_test_value = 'raise'

from attention_mechanism_rectangular_multichannel import (
    SelectiveAttentionMechanism)
from common.model import Model
from rnn import RNN


class RATM(Model):
    @property
    def _grads(self):
        if self.__grads is None:
            self.__grads = T.grad(self._cost, self.params)
        return self.__grads

    def __init__(self, name, imsize, patchsize, nhid,
                 numpy_rng, eps, hids_scale,
                 feature_network=None, input_feature_layer_name=None,
                 metric_feature_layer_name=None,
                 nchannels=1, weight_decay=0.):
        # CALL PARENT CONSTRUCTOR TO SETUP CONVENIENCE FUNCTIONS
        # (SAVE/LOAD, ...)
        super(RATM, self).__init__(name=name)
        self.imsize = imsize
        assert len(patchsize) == 2
        self.patchsize = patchsize
        self.nhid = nhid
        self.numpy_rng = numpy_rng
        self.eps = eps
        self.hids_scale = hids_scale
        self.nchannels = nchannels
        self.weight_decay = weight_decay
        assert hasattr(feature_network, 'forward')
        assert hasattr(feature_network, 'load')
        self.feature_network = feature_network
        self.input_feature_layer_name = input_feature_layer_name
        assert (self.input_feature_layer_name in
                self.feature_network.layers.keys())
        self.metric_feature_layer_name = metric_feature_layer_name
        assert (self.metric_feature_layer_name in
                self.feature_network.layers.keys())
        # TODO: remove this constraint, if everything else works
        assert (
            self.feature_network.layers.keys().index(
                self.metric_feature_layer_name) >
            self.feature_network.layers.keys().index(
                self.input_feature_layer_name))

        ftensor5 = T.TensorType(theano.config.floatX, (False,) * 5)
        self.inputs = ftensor5(name='inputs')
        self.inputs.tag.test_value = numpy_rng.randn(
            16, 5, nchannels, imsize[0], imsize[1]).astype(np.float32)
        self.targets = T.ftensor3(name='targets')
        self.targets.tag.test_value = numpy_rng.randn(
            16, 5, 4).astype(np.float32)
        self.masks = T.fmatrix(name='masks')
        self.masks.tag.test_value = np.ones((16, 5), dtype=np.float32)

        self.batchsize = self.inputs.shape[0]
        self.nframes = self.inputs.shape[1]

        # shuffle axis, such that time axis is first
        self.inputs_frames = self.inputs.transpose(1, 0, 2, 3, 4)
        self.targets_frames = self.targets.transpose(1, 0, 2)
        self.masks_frames = self.masks.T

        self.attention_mechanism = SelectiveAttentionMechanism(
            imsize=imsize, patchsize=patchsize, eps=self.eps,
            nchannels=nchannels)

        self.targets_widthheight = (self.targets_frames[:, :, 1::2] -
                                    self.targets_frames[:, :, ::2])
        self.targets_XYs = (self.targets_frames[:, :, 1::2] +
                            self.targets_frames[:, :, ::2]) / 2.

        self.targets_centers_widthheight = T.concatenate((
            self.targets_XYs, self.targets_widthheight), axis=2)

        self.nin = self.feature_network.layers[
            self.input_feature_layer_name].outputs_shape[1]
        self.rnn = RNN(nin=self.nin, nout=10, nhid=self.nhid,
                       numpy_rng=self.numpy_rng, scale=hids_scale)

        self.wread = theano.shared(
            numpy_rng.uniform(
                low=-.001, high=.001, size=(self.nhid, 7)
            ).astype(np.float32), name='wread')

        self.targets_params = T.concatenate((
            # center x,y
            self.targets_centers_widthheight[
                :, :, :2] / np.array(((imsize[::-1],),), dtype=np.float32),
            # std x
            (self.targets_centers_widthheight[:, :, 2] /
             patchsize[1]).dimshuffle(0, 1, 'x'),
            # stride x
            np.float32(1.5) * (self.targets_centers_widthheight[:, :, 2] /
                               imsize[1]).dimshuffle(0, 1, 'x'),
            # gamma (unused)
            T.ones((self.nframes, self.batchsize, 1)),
            # std y
            (self.targets_centers_widthheight[:, :, 3] /
             patchsize[0]).dimshuffle(0, 1, 'x'),
            # stride y
            np.float32(1.5) * (self.targets_centers_widthheight[:, :, 3] /
                               imsize[0]).dimshuffle(0, 1, 'x'),
        ), axis=2)

        self.targets_params_reshape = self.targets_params.reshape((
            self.nframes * self.batchsize, 7
        ))

        (self.targets_patches,
         _, _, _, _) = self.attention_mechanism.build_read_graph(
            images_var=self.inputs_frames.reshape((
                self.nframes * self.batchsize, self.nchannels,
                self.imsize[0], self.imsize[1])),
            attention_acts=self.targets_params_reshape)

        self.targets_features = self.feature_network.forward_from_to(
            self.targets_patches,
            to_layer_name=self.metric_feature_layer_name
        )
        self.targets_features = self.targets_features.reshape((
            self.nframes, self.batchsize,
            T.prod(self.targets_features.shape[1:])))

        self.bread_init = T.concatenate((
            # center x,y
            self.targets_centers_widthheight[
                0, :, :2] / np.array((imsize[::-1],), dtype=np.float32),
            # std x
            (self.targets_centers_widthheight[0, :, 2] /
             patchsize[1]).dimshuffle(0, 'x'),
            # stride x
            np.float32(1.5) * (self.targets_centers_widthheight[0, :, 2] /
                               imsize[1]).dimshuffle(
                0, 'x'),
            # gamma (unused)
            T.ones((self.batchsize, 1)),
            # std y
            (self.targets_centers_widthheight[0, :, 3] /
             patchsize[0]).dimshuffle(0, 'x'),
            # stride y
            np.float32(1.5) * (self.targets_centers_widthheight[0, :, 3] /
                               imsize[0]).dimshuffle(
                0, 'x'),
        ), axis=1)

        self.params = [self.wread]  # , self.bread_init_factors]
        self.params.extend(self.rnn.params)
        # we're not using the rnn output layer, so remove params
        self.params.remove(self.rnn.wout)
        self.params.remove(self.rnn.bout)

        def step(x_t, h_tm1, bread, wread):
            (patches_t, window_params_t, muX, muY,
             gX, gY) = self.get_input_patches(
                x_t, h_tm1, wread, bread)
            features_t = self.feature_network.forward_from_to(
                patches_t,
                from_layer_name=self.feature_network.layers.keys()[0],
                to_layer_name=self.input_feature_layer_name)
            h_t, o_t = self.rnn.step(features_t, h_tm1)
            h_t_norm = T.sqrt(T.sum(h_t**2, axis=-1))
            return (h_t, window_params_t, patches_t, features_t,
                    window_params_t, muX, muY, gX, gY, h_t_norm)

        (self.hiddens, breads, self.patches, self.features,
         self.window_params, muX, muY, gX, gY, h_t_norms), self.updates = theano.scan(
             fn=step,
             sequences=self.inputs_frames,
             outputs_info=[
                 T.zeros((self.batchsize, self.nhid),
                         dtype=theano.config.floatX),
                 self.bread_init,
                 None, None, None, None,
                 None, None, None, None],
            non_sequences=[self.wread])

        # vector containing corner  mus of window, in order x1, x2, y1, y2
        self._attention_mus = T.concatenate((
            muX[:, :, 0].dimshuffle(0, 1, 'x'),
            muX[:, :, -1].dimshuffle(0, 1, 'x'),
            muY[:, :, 0].dimshuffle(0, 1, 'x'),
            muY[:, :, -1].dimshuffle(0, 1, 'x')), axis=2)
        self._attention_gs = T.concatenate((
            gX.dimshuffle(0, 1, 'x'),
            gY.dimshuffle(0, 1, 'x')), axis=2)

        # get index of layer after feature layer
        after_feat_layer_idx = self.feature_network.layers.keys().index(
            self.input_feature_layer_name) + 1

        self.attention_features = self.feature_network.forward_from_to(
            self.features.reshape((T.prod(self.features.shape[:2]),
                                   self.features.shape[2])),
            from_layer_name=self.feature_network.layers.keys()[
                after_feat_layer_idx],
            to_layer_name=self.metric_feature_layer_name
        ).reshape((
            self.nframes, self.batchsize, self.targets_features.shape[2]
        ))

        self._stepcosts = T.mean((
            self.targets_features - self.attention_features)**2, axis=2)

        self._dists = self._stepcosts

        # normalize mask to sum up to 1 for each sequence, to give equal
        # contribution to long and short sequences
        self._stepcosts_masked = (
            self._stepcosts * self.masks_frames) / T.sum(
            self.masks_frames, axis=0, keepdims=True)

        self._cost = (
            T.mean(self._stepcosts_masked) +
            self.weight_decay * (
                T.mean(self.rnn.win**2) + T.mean(self.wread**2)
            )
        )

        # grads graph will be built when first accessed
        self.__grads = None

        target_centers_widthheight = T.ftensor3('target_centers_widthheight')
        target_centers_widthheight.tag.test_value = numpy_rng.rand(
            16, 5, 4).astype(np.float32)

        print "compiling get_all_patches_and_windows..."
        self.get_all_patches_and_windows = theano.function(
            [self.inputs, target_centers_widthheight],
            [self.patches, self.window_params],
            givens={
                self.targets_centers_widthheight:
                    target_centers_widthheight.dimshuffle(1, 0, 2)})
        print "done (with compiling get_all_patches_and_windows)"

        print "compiling get_all_patches_and_windows_and_dists..."
        self.get_all_patches_and_windows_and_probs = theano.function(
            [self.inputs, target_centers_widthheight],
            [self.patches, self.window_params, self._dists],
            givens={
                self.targets_centers_widthheight:
                    target_centers_widthheight.dimshuffle(1, 0, 2)})
        print "done (with compiling get_all_patches_and_windows_and_dists)"

        self.get_bbs = theano.function(
            [self.inputs, target_centers_widthheight],
            self._attention_mus,
            givens={
                self.targets_centers_widthheight:
                    target_centers_widthheight.dimshuffle(1, 0, 2)})

    def get_input_patches(self, x, h, wread, bread):
        window_params = T.dot(h, wread) + bread
        patches, muX, muY, gX, gY = self.attention_mechanism.build_read_graph(
            images_var=x, attention_acts=window_params)
        return (patches, window_params, muX, muY, gX, gY)


# vim: set ts=4 sw=4 sts=4 expandtab:
