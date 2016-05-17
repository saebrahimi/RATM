#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np

ious = []
for i in range(25):
    ious.append(np.load('kth_{0:02d}left_out_IoU_test.npy'.format(i + 1)))

print 'avg. IoU: {}'.format(np.array(ious).mean())


# vim: set ts=4 sw=4 sts=4 expandtab:
