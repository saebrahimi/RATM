#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams

from attention_rnn import RATM
from convnet_eth_inria_data.cnn import HumanConvNet
sys.path.append('..')
from common.bbox import BoundingBox
from data_preparation.kth.kth_data import KTHDataProvider

# the leave-one-out sample for this run
parser = argparse.ArgumentParser()
parser.add_argument('--test-person', type=int, required=True)

args = parser.parse_args()

theano_rng = RandomStreams(1)
patchsize = (28, 28)
imsize = (120, 160)
nhid = 32
batchsize = 16
loadsize = 128
input_feature_layer_name = 'reshape1'
metric_feature_layer_name = 'relu2'

plt.gray()

weight_decay = np.float32(.2)

numpy_rng = np.random.RandomState(1)

print 'fetching test set...'
test_data_provider = KTHDataProvider(
    numpy_rng=numpy_rng,
    frames_dir='/data/lisatmp3/michals/data/KTH/frames',
    pkl_file='kth.pkl',
    bbox_file='/data/lisatmp3/michals/data/KTH/KTHBoundingBoxInfo.txt',
    persons=[args.test_person], actions=('jogging', 'running', 'walking'))
test_data = test_data_provider.get_batch()

print 'loading pretrained CNN...'
feature_network = HumanConvNet(
    name='Person CNN', nout=2, numpy_rng=numpy_rng,
    theano_rng=theano_rng, batchsize=batchsize)
feature_network.load('convnet_eth_inria_data/human_convnet_val_best.h5')
feature_network.mode.set_value(np.uint8(1))

print "instantiating model..."
model = RATM(name='RATM', imsize=imsize,
             patchsize=patchsize, nhid=nhid,
             numpy_rng=numpy_rng, eps=1e-4,
             hids_scale=1.,
             feature_network=feature_network,
             input_feature_layer_name=input_feature_layer_name,
             metric_feature_layer_name=metric_feature_layer_name,
             nchannels=1,
             weight_decay=weight_decay)
print "done (with instantiating model)"

model.load(
    'attention_model_kth_{0:02d}left_out_val_best.h5'.format(
        args.test_person))


def compute_avg_IoU(inputs, targets, masks):
    bbs = targets
    vids = inputs
    max_nframes = np.max(np.where(masks > .5)[1])

    N = vids.shape[0]

    Xs = (bbs[:, :, 1::2] + bbs[:, :, ::2]) / 2.

    # left, right, top, bottom (w/h = r-l, b-t)
    width_height = bbs[:, :, 1::2] - bbs[:, :, ::2]
    vids = vids.astype(np.float32)

    Xs_widthheight = np.concatenate((
        Xs, width_height), axis=2)

    pred_bbs = model.get_bbs(
        vids[:, :max_nframes],
        Xs_widthheight[:, :max_nframes].astype(np.float32)
    ).transpose(1, 0, 2)

    mean_IoU = 0.
    for n in range(N):
        clip_mean_IoU = 0.
        nframes = np.max(np.where(masks[n] > .5)[0])
        for t in range(nframes):
            if masks[n, t] < .5:
                continue
            dx_ = pred_bbs[n, t, 1] - pred_bbs[n, t, 0]
            dy_ = pred_bbs[n, t, 3] - pred_bbs[n, t, 2]
            pred_bb = BoundingBox(
                x=pred_bbs[n, t, 0] + .25/1.5 * dx_,
                y=pred_bbs[n, t, 2] + .25/1.5 * dy_,
                dx=1. / 1.5 * dx_,
                dy=1. / 1.5 * dy_)

            x = bbs[n, t, 0]
            y = bbs[n, t, 2]
            dx = (bbs[n, t, 1] - bbs[n, t, 0])
            dy = (bbs[n, t, 3] - bbs[n, t, 2])
            # scale up by 1.5 (provided ground truth bbs are too tight, and
            # they are scaled up in the model as well)
            gt_bb = BoundingBox(
                x=x,
                y=y,
                dx=dx,
                dy=dy
            )
            clip_mean_IoU += pred_bb.overlap(gt_bb) / nframes
        mean_IoU += clip_mean_IoU / N
    return mean_IoU


performance = compute_avg_IoU(**test_data)

print 'IoU: ', performance
np.save('kth_{0:02d}left_out_IoU_test.npy'.format(args.test_person),
        performance)

# vim: set ts=4 sw=4 sts=4 expandtab:
