#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T

from imtoolbox import autocrop


def corrupt(x, corruption_type, corruption_level, theano_rng):
    if corruption_type == 'zeromask':
        return x * theano_rng.binomial(
            size=x.shape, n=1, p=1.0 - corruption_level,
            dtype=theano.config.floatX)
    elif corruption_type == 'gaussian':
        return x * theano_rng.normal(
            size=x.shape, avg=0, std=corruption_level + .0000001,
            dtype=theano.config.floatX) + x
    elif corruption_type is None:
        return x
    else:
        raise ValueError('unknown corruption type')


def logodds(x):
    clip_x = T.clip(x, 1e-5, 1-1e-5)
    return T.log(clip_x / (1. - clip_x))


def mse(x, x_hat, axis):
    rval = T.sum((x - x_hat)**2, axis=-1)
    if axis is not None:
        rval = T.mean(rval, axis=axis)
    return rval


def mad(x, x_hat, axis):
    rval = T.mean(abs(x - x_hat), axis=-1)
    if axis is not None:
        rval = T.mean(rval, axis=axis)
    return rval


def symmetric_mse(x, x_hat, y, y_hat, axis=None):
    rval = T.sum(
        .5 * (x - x_hat)**2 + .5 * (y - y_hat)**2, axis=-1)
    if axis is not None:
        rval = T.mean(rval, axis=axis)
    return rval


def cross_entropy(x, x_hat, axis):
    rval = - T.mean((x * T.log(x_hat) + (1. - x) * T.log(1. - x_hat)), axis=-1)
    if axis is not None:
        rval = T.mean(rval, axis=axis)
    return rval


def symmetric_cross_entropy(x, x_hat, y, y_hat, axis):
    rval = - T.sum(
        .5 * (x * T.log(x_hat) +
              (1. - x) * T.log(1. - x_hat)) +
        .5 * (y * T.log(y_hat) +
              (1. - y) * T.log(1. - y_hat)),
        axis=-1)
    if axis is not None:
        rval = T.mean(rval, axis=axis)
    return rval


def make_debug_html(target_folder, ims_and_captions):
    """
    Creates a debug html page
    Args
    ----
        target_folder: where to save the html page
        ims_and_captions: list of tuples containing im_filenames and caption
    """
    # create target dir if not already existing
    imdir = os.path.join(target_folder, 'img')
    os.system('mkdir -p {0}'.format(imdir))
    print imdir

    # load htmlhtml  template
    fid = file('template.html')
    template_string = fid.read()
    fid.close()

    # define template for image display element
    im_cap_str = "<p><h2>{1}</h2><br /><img src='{0}' /></p>"

    # loop over images
    tmp_str = ""
    for im_path, cap in ims_and_captions:
        # image target path
        im_path_base = os.path.basename(im_path)
        im_target_path = os.path.join(imdir, im_path_base)
        im_rel_path = os.path.join('img', im_path_base)
        if os.path.splitext(im_path_base)[-1] in ('.png',):
            # autocrop image
            im = plt.imread(im_path)
            im = autocrop(im)
            plt.imsave(im_target_path, im)
        else:
            # just copy to target folder
            shutil.copyfile(im_path, im_target_path)

        # generate code for embedding img in html page
        tmp_str += im_cap_str.format(im_rel_path, cap)

    # write html page
    fid = file('{0}/index.html'.format(target_folder), 'w')
    fid.write(template_string.format(body=tmp_str, head='Debug overview'))

    # copy stylesheet of the template
    shutil.copyfile('template.css', os.path.join(
        target_folder, 'template.css'))
    fid.close()


def onehot(x, numclasses=None):
    """ Convert integer encoding for class-labels (starting with 0 !)
        to one-hot encoding.

        If numclasses (the number of classes) is not provided, it is assumed
        to be equal to the largest class index occuring in the labels-array + 1.
        The output is an array who's shape is the shape of the input array plus
        an extra dimension, containing the 'one-hot'-encoded labels.
    """
    if x.shape == ():
        x = x[np.newaxis]
    if numclasses is None:
        numclasses = x.max() + 1
    result = np.zeros(list(x.shape) + [numclasses])
    z = np.zeros(x.shape)
    for c in range(numclasses):
        z *= 0
        z[np.where(x == c)] = 1
        result[..., c] += z
    return result


def random_orthogonal_mat(shp, numpy_rng):
    tmp = numpy_rng.normal(loc=0, scale=1, size=shp)
    u, _, v = np.linalg.svd(tmp, full_matrices=False)
    if u.shape == shp:
        return np.float32(u)
    else:
        return np.float32(v[:shp[0], :shp[1]])


def random_sparse_mat(shp, sparsity, scale, numpy_rng):
    n_nonzero = int(sparsity * shp[0])
    w = np.zeros(shp, dtype=np.float32)
    for c in range(shp[1]):
        idx = numpy_rng.permutation(shp[0])[:n_nonzero]
        val = numpy_rng.uniform(low=-scale, high=scale, size=n_nonzero)
        w[idx, c] = val
    return w

# vim: set ts=4 sw=4 sts=4 expandtab:
