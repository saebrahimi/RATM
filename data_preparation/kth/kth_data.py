#!/usr/bin/python
# -*- coding: utf-8 -*-

import cPickle as pickle
import os

import numpy as np
from PIL import Image
import tables


class KTHDataProvider(object):

    classes = ('boxing', 'handclapping', 'handwaving',
               'jogging', 'running', 'walking')
    persons = range(1, 26)

    def __init__(self, numpy_rng, frames_dir=None,
                 bbox_file=None, pkl_file=None,
                 persons=(), actions=(),
                 minlen=None, maxlen=None,
                 batchsize=None):
        self.numpy_rng = numpy_rng
        self.pkl_file = pkl_file
        self.frames_dir = frames_dir
        self.bbox_file = bbox_file
        self.minlen = minlen
        self.maxlen = maxlen
        self.batchsize = batchsize
        if os.path.exists(self.pkl_file):
            self.load_from_pkl_file()
        else:
            self.preload_videos()
        self.persons = persons
        self.actions = actions

        # filter by person and action
        person_indices = []
        action_indices = []
        for p in self.persons:
            person_indices.extend(np.where(self.person_ids == p)[0])
        for a in self.actions:
            action_indices.extend(np.where(
                self.action_labels == self.classes.index(a))[0])
        self.indices = np.array(list(set(person_indices).intersection(
            set(action_indices))))
        if batchsize is not None:
            self.n_batches = len(self.indices) / batchsize

    def load_from_pkl_file(self):
        tmp = pickle.load(file(self.pkl_file))
        self.bboxes = tmp.bboxes
        self.person_ids = tmp.person_ids
        self.scenario_ids = tmp.scenario_ids
        self.sequence_ids = tmp.sequence_ids
        self.action_labels = tmp.action_labels
        self.start_indices = tmp.start_indices
        self.end_indices = tmp.end_indices
        self.folder_names = tmp.folder_names

    def preload_videos(self):
        with file(self.bbox_file) as fid:
            lines = fid.readlines()
        person_labels = []
        scenario_indices = []
        action_labels = []
        sequence_indices = []
        bboxes = []
        frames = []
        folder_names = []
        start_indices = []
        end_indices = []
        for l, line in enumerate(lines[4:]):
            print l + 1, '/', len(lines)
            line_split = line.rstrip().split(' ', 6)

            try:
                bboxes.append(np.fromstring(
                    line_split[6], dtype=np.int32, sep=' ').reshape(-1, 4))
                # FIX COORDINATE ORDER FROM y1,x1,y2,x2 TO x1,x2,y1,y2
                bboxes[-1] = np.hstack((
                    bboxes[-1][:, 1, None],
                    bboxes[-1][:, 3, None],
                    bboxes[-1][:, 0, None],
                    bboxes[-1][:, 2, None]))
                sequence_indices.append(int(line_split[3]))
                action_labels.append(int(line_split[2]) - 1)
                scenario_indices.append(int(line_split[1]))
                person_labels.append(int(line_split[0]))
                start_indices.append(int(line_split[4]))
                end_indices.append(int(line_split[5]))
                folder_names.append(os.path.join(
                    self.frames_dir,
                    'person{0:02d}_{1}_d{2}_uncomp'.format(
                        person_labels[-1],
                        self.classes[action_labels[-1]],
                        scenario_indices[-1])))
            except:
                # skip missing videos
                continue
        self.bboxes = np.array(bboxes)

        # print 'done'
        self.person_ids = np.array(person_labels, dtype=np.int8)
        self.scenario_ids = np.array(scenario_indices, dtype=np.int8)
        self.sequence_ids = np.array(sequence_indices, dtype=np.int8)
        self.action_labels = np.array(action_labels, dtype=np.int8)
        self.start_indices = start_indices
        self.end_indices = end_indices
        self.folder_names = folder_names
        print 'dumping pkl file...'
        pickle.dump(self, file(self.pkl_file, 'w'))
        print 'done'

    def get_batch(self, return_all=False):
        shuffled_indices = self.indices[self.numpy_rng.permutation(
            len(self.indices))]
        frames = []
        start_offsets = []
        batch_indices = []
        for idx in shuffled_indices:
            if self.batchsize is None or return_all:
                start = self.start_indices[idx]
                end = self.end_indices[idx]
            else:
                if self.end_indices[idx] - self.start_indices[idx] < self.minlen:
                    continue
                start = self.numpy_rng.randint(
                    self.start_indices[idx], self.end_indices[idx] - self.minlen)
                end = start + self.numpy_rng.randint(
                    self.minlen, min(
                        self.maxlen, self.end_indices[idx] - start) + 1)
            start_offsets.append(start - self.start_indices[idx])
            frames.append(np.array([np.asarray(Image.open(os.path.join(
                self.folder_names[idx],
                '{0:03d}.png'.format(frame_no))).convert(
                'L')) for frame_no in range(start, end)]))
            batch_indices.append(idx)
            if not return_all and (self.batchsize is not None and
                                   len(frames) == self.batchsize):
                break
        else:
            if self.batchsize is not None and not return_all:
                raise Exception((
                    'Couldn\'t get enough videos with the specified minimum '
                    'length ({0})').format(self.minlen))

        # get length of longest vid in batch, generate masks and pad videos
        maxlen = 0
        for v in frames:
            if maxlen < len(v):
                maxlen = len(v)
        # print 'max length in current batch: ', maxlen
        masks = np.zeros((len(frames), maxlen), dtype=np.float32)
        vids = np.zeros((len(frames), maxlen, 120, 160, 1), dtype=np.float32)
        bboxes = np.zeros((len(frames), maxlen, 4), dtype=np.float32)
        for i, v in enumerate(frames):
            vids[i, :len(v)] = v[..., None]
            masks[i, :len(v)] = 1.
            bboxes[i, :len(v)] = self.bboxes[batch_indices[i]][
                start_offsets[i]:start_offsets[i] + len(v)]
        return {'inputs': vids.transpose(0, 1, 4, 2, 3), 'masks': masks,
                'targets': bboxes}


if __name__ == '__main__':
    numpy_rng = np.random.RandomState(1)
    dataset = KTHDataProvider(
        numpy_rng=numpy_rng,
        frames_dir='./frames', bbox_file='./KTHBoundingBoxInfo.txt',
        pkl_file='./kth_w_bboxes.pkl', persons=range(1, 26),
        actions=('jogging', 'running', 'walking'), maxlen=15)
    batch = dataset.get_batch(batchsize=10)
    import matplotlib.pyplot as plt
    N, T = batch['inputs'].shape[:2]
    for i in range(N):
        for t in range(T):
            plt.subplot(N, T, i * T + t + 1)
            if batch['masks'][i, t]:
                plt.imshow(batch['inputs'][i, t])
                x1, x2, y1, y2 = batch['bboxes'][i, t]
                plt.scatter([x1, x2, x1, x2], [y1, y1, y2, y2])
            plt.axis('off')
    plt.savefig('test annotations.png')

# vim: set ts=4 sw=4 sts=4 expandtab:
