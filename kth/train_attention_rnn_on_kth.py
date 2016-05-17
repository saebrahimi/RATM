#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
os.system('mkdir -p tmp vis')
import sys
sys.path.append('..')

import matplotlib
matplotlib.use('agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
numpy_rng = np.random.RandomState(1)
import tables
from theano.tensor.shared_randomstreams import RandomStreams
import theano
theano.config.compute_test_value = 'ignore'

from attention_rnn import RATM
from convnet_eth_inria_data.cnn import HumanConvNet
from common.bbox import BoundingBox
from common.trainer import SGD_Trainer
from data_preparation.kth.kth_data import KTHDataProvider

numpy_rng = np.random.RandomState(1)

# the leave-one-out sample for this run
parser = argparse.ArgumentParser()
parser.add_argument('--test-person', type=int, required=True)

args = parser.parse_args()

persons = range(1, 26)
assert args.test_person in persons
# remove the leave-out subject
persons.remove(args.test_person)
# pick a random validation subject and remove it from the training set
val_person = persons[numpy_rng.randint(len(persons))]
persons.remove(val_person)
train_persons = persons

plt.gray()
theano_rng = RandomStreams(1)
patchsize = (28, 28)
imsize = (120, 160)
nhid = 32
batchsize = 16
loadsize = 128
input_feature_layer_name = 'reshape1'
metric_feature_layer_name = 'relu2'

minlen = 5
maxlen = 5

weight_decay = np.float32(.2)
gradient_clip_threshold = 1.

train_data_provider = KTHDataProvider(
    batchsize=loadsize, minlen=minlen, maxlen=maxlen, numpy_rng=numpy_rng,
    frames_dir='../data_preparation/kth/frames',
    pkl_file='kth.pkl',
    bbox_file='../data_preparation/kth/KTHBoundingBoxInfo.txt',
    persons=train_persons, actions=('jogging', 'running', 'walking'))
print 'fetching 100 samples from training set for approx of train IoUs...'
train_data = train_data_provider.get_batch(return_all=True)

perm = numpy_rng.permutation(train_data['inputs'].shape[0])[:100]

train_data['inputs'] = train_data['inputs'][perm]
train_data['targets'] = train_data['targets'][perm]
train_data['masks'] = train_data['masks'][perm]

print 'fetching validation set...'
val_data_provider = KTHDataProvider(
    numpy_rng=numpy_rng,
    frames_dir='../data_preparation/kth/frames',
    pkl_file='kth.pkl',
    bbox_file='../data_preparation/kth/KTHBoundingBoxInfo.txt',
    persons=[val_person], actions=('jogging', 'running', 'walking'))
val_data = val_data_provider.get_batch()
print val_data['inputs'].shape

print 'fetching test set...'
test_data_provider = KTHDataProvider(
    numpy_rng=numpy_rng,
    frames_dir='../data_preparation/kth/frames',
    pkl_file='kth.pkl',
    bbox_file='../data_preparation/kth/KTHBoundingBoxInfo.txt',
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


def visualize(fname):
    n = 5
    idx = numpy_rng.permutation(len(val_data['inputs']))[:n]
    val_vids = val_data['inputs'][idx]
    val_bbs = val_data['targets'][idx]
    val_masks = val_data['masks'][idx]

    val_Xs = (val_bbs[:, :, 1::2] + val_bbs[:, :, ::2]) / 2.

    # left, right, top, bottom (w/h = r-l, b-t)
    val_width_height = val_bbs[:, :, 1::2] - val_bbs[:, :, ::2]
    val_vids = val_vids.astype(np.float32)

    val_Xs_widthheight = np.concatenate((
        val_Xs[:n, :], val_width_height[:n, :]), axis=2)

    patches, windows, dists = model.get_all_patches_and_windows_and_probs(
        val_vids[:n],
        val_Xs_widthheight.astype(np.float32))

    plt.clf()
    nsteps = patches.shape[0]
    # Visualize patches
    for i in range(n):
        for t in range(nsteps):
            if val_masks[i, t] < .5:
                continue
            ax = plt.subplot(2 * n, nsteps, i * 2 * nsteps + t + 1)
            # move channel axis to end
            im = val_vids[i, t, 0]
            plt.imshow(im.astype(np.uint8))
            (gX, gY, sigmaX, sigmaY, deltaX, deltaY, gammas,
             muX, muY) = model.attention_mechanism.get_window_params_numpy(
                 windows[t])
            ax.add_patch(Rectangle(
                (muX[i, 0, 0], muY[i, 0, 0]), muX[i, -1, 0] - muX[i, 0, 0],
                muY[i, -1, 0] - muY[i, 0, 0], linewidth=.7, edgecolor='r',
                fill=False))
            ax.add_patch(Rectangle(
                (val_bbs[i, t, 0], val_bbs[i, t, 2]),
                val_width_height[i, t, 0], val_width_height[i, t, 1],
                linewidth=.7, edgecolor='g', fill=False))
            plt.imshow(im.astype(np.uint8))
            plt.axis('off')
            plt.subplot(2 * n, nsteps, i * 2 * nsteps + t + 1 + nsteps)
            plt.imshow(patches[t, i, 0].reshape(
                patchsize[0], patchsize[1]).astype(np.uint8))
            plt.title('{0:.2f}'.format(dists[t, i]), fontsize=1)
            plt.axis('off')
    plt.savefig(fname, dpi=500)

print 'visualizing...'
visualize('vis/tracking_kth_{0:02d}left_out.pdf'.format(
    args.test_person))
print 'done (with visualizing)'


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
            pred_bb = BoundingBox(
                x=pred_bbs[n, t, 0],
                y=pred_bbs[n, t, 2],
                dx=pred_bbs[n, t, 1] - pred_bbs[n, t, 0],
                dy=pred_bbs[n, t, 3] - pred_bbs[n, t, 2])

            x = bbs[n, t, 0]
            y = bbs[n, t, 2]
            dx = (bbs[n, t, 1] - bbs[n, t, 0])
            dy = (bbs[n, t, 3] - bbs[n, t, 2])
            # scale up by 1.5 (provided ground truth bbs are too tight, and
            # they are scaled up in the model as well)
            gt_bb = BoundingBox(
                x=x - dx / 4.,
                y=y - dy / 4.,
                dx=dx * 1.5,
                dy=dy * 1.5
            )
            clip_mean_IoU += pred_bb.overlap(gt_bb) / nframes
        mean_IoU += clip_mean_IoU / N
    return mean_IoU


def compute_avg_IoU_batchwise(inputs, targets, masks):
    n = len(inputs)

    batchsize = 100
    n_batches = int(np.ceil(float(n) / batchsize))
    print 'n_batches: ', n_batches
    mean_IoU = 0.
    for i in range(n_batches):
        start = i * batchsize
        end = min(n, (i + 1) * batchsize)
        print 'batch {0}/{1} (size {2})'.format(
            i + 1, n_batches, end - start)
        mean_IoU += float(end - start) / n * compute_avg_IoU(
            inputs=inputs[start:end],
            targets=targets[start:end],
            masks=masks[start:end])
    return mean_IoU


train_IoUs = [compute_avg_IoU(**train_data)]
val_IoUs = [compute_avg_IoU(**val_data)]
test_IoUs = [compute_avg_IoU(**test_data)]

best_val_IoU = val_IoUs[0]
best_val_IoU_epoch = 0

model.learningrate_modifiers = {'wrnn': np.float32(1)}

print "instantiating trainer..."
trainer = SGD_Trainer(
    model=model,
    learningrate=.001,
    momentum=.0, gradient_clip_threshold=gradient_clip_threshold,
    batchsize=batchsize, monitor_update_weight_norm_ratio=True,
    batch_fn=train_data_provider.get_batch,
    numloads=train_data_provider.n_batches)
print "done (with instantiating trainer)"

print "starting training..."
for epoch in range(500):
    # increase maximum length of training sequences by 1 every 20 epochs
    train_data_provider.maxlen = min(epoch // 20 + maxlen, 30)
    trainer.step()

    # compute training meanIoU
    print 'computing mean IoUs...'
    train_IoUs.append(compute_avg_IoU(**train_data))
    # compute validation meanIoU
    val_IoUs.append(compute_avg_IoU(**val_data))
    # compute test meanIoU
    test_IoUs.append(compute_avg_IoU(**test_data))

    with tables.openFile('ious_kth{0:02d}left_out.h5'.format(args.test_person),
                         'w') as h5file:
        h5file.createArray(h5file.root, 'train_IoUs', np.array(train_IoUs))
        h5file.createArray(h5file.root, 'val_IoUs', np.array(val_IoUs))
        h5file.createArray(h5file.root, 'test_IoUs', np.array(test_IoUs))
        h5file.createArray(h5file.root, 'trainer_costs', trainer.costs)

    if best_val_IoU < val_IoUs[-1]:
        best_val_IoU = val_IoUs[-1]
        best_val_IoU_epoch = epoch + 1
        model.save(
            'attention_model_kth_{0:02d}left_out_val_best.h5'.format(
                args.test_person))

    print 'train IoU: {0:f}'.format(train_IoUs[-1])
    print 'val IoU: {0:f} (best so far {1} in epoch {2})'.format(
        val_IoUs[-1], best_val_IoU, best_val_IoU_epoch)
    print 'test IoU: {0:f}'.format(test_IoUs[-1])

    plt.clf()
    plt.plot(range(epoch + 2), train_IoUs, c='g',
             label='Avg. IoU on Train')
    plt.plot(range(epoch + 2), val_IoUs, c='b',
             label='Avg. IoU on Val (person {0})'.format(val_person))
    plt.scatter([best_val_IoU_epoch], [best_val_IoU],
                marker='*', c='b')
    plt.xlabel('epoch')
    plt.ylabel('avg. IoU')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.xlim(0, epoch + 2)
    plt.savefig('vis/train_val_IoUs_{0:02d}left_out.png'.format(
        args.test_person))
    plt.plot(range(epoch + 2), test_IoUs, c='r',
             label='Avg. IoU on Test (person {0})'.format(args.test_person))
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.,
               fontsize=9)
    plt.xlim(0, epoch + 2)
    plt.savefig('vis/train_val_test_IoUs{0:02d}left_out.png'.format(
        args.test_person))

    plt.clf()
    plt.plot(np.arange(1, epoch + 2), trainer.costs, label='Cost on Train')
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig('vis/train_loss_plot_{0:02d}left_out.png'.format(
        args.test_person))

    model.save('attention_model_kth_trainlen{0:03d}_{1:02d}left_out.h5'.format(
        train_data_provider.maxlen, args.test_person))
    model.save('attention_model_kth_{0:02d}left_out.h5'.format(
        args.test_person))
    if (epoch + 1) % 1 == 0:
        print 'visualizing...'
        visualize('vis/tracking_kth_{0:02d}left_out.pdf'.format(
            args.test_person))
        print 'done (with visualizing)'
print "done (with training)"

# vim: set ts=4 sw=4 sts=4 expandtab:
