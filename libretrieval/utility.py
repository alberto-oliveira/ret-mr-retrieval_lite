#/usr/bin/env python
# -*- coding: utf-8 -*-

"""
objret.utility.py

Several utility functions.

"""

import sys
import os
import traceback
import errno
import configparser

import cv2
import numpy as np

#
def safe_create_dir(dir):
    """ Safely creates dir, checking if it already exists.

    Creates any parent directory necessary. Raises exception
    if it fails to create dir for any reason other than the
    directory already existing.

    :param dir: of the directory to be created
    """


    try:
        os.makedirs(dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def cfgloader(cfgfile):
    """

    Loads a config file into a ConfigParser object.

    :param cfgfile: path to the configuration file.
    :return: ConfigParser object.
    """

    if os.path.isfile(cfgfile):
        config = configparser.ConfigParser()
        config.read(cfgfile)
    else:
        raise ValueError("<{0:s}> is not a file".format(cfgfile))

    return config


def keypoints_to_points(keyps):
    """ Given a list of keypoints in OpenCV's format, returns an with the xy coordinates of
        each keypoint.

    :param keyps: keypoints to be converted.

    """

    ptlist = []
    for kp in keyps:
        ptlist.append(kp.pt)

    return np.array(ptlist, dtype=np.float32)


def load_feature_indexing(findexingpath):
    """ Loads a feature indexing file. This files relates the features from a set to their source
        images, through indexing.

    Each row of this file contains three information:
    <img> <nfeat> <topi>

    Where <img> is the name of the source image, nfeat is the number of features extracted from <img> and
    <topi> is the index of the top feature that do not belong to that image. To compute the range of features
    from <img>, we start from index <topi> - <nfeat>, until <topi>. For example, if we have:

    image_X 455 1512

    it means that image_X has 455 features, in the range of indexes [1512-455 1511] or [1057 1511]

    Note: this format could be simplified to only use the topi, since we know features start indexing from 0.

    :param findexingpath: path to the feature indexing file.

    :return: numpy array containing the namelist of the database;
    :return: numpy array indexing features to image indextable[i] = X means that feature[i] belongs to
                         image X;
    :return: numpy array containing the number of features of each image;
    :return: numpy array containing the top indexes for each image;

    """

    try:

        idxdt = dict(names=('name', 'nfeat', 'topidx'), formats=('S100', 'i32', 'i32'))
        dbfeatindex = np.loadtxt(findexingpath, dtype=idxdt)
        nametable = np.array(dbfeatindex['name'])
        featntable = np.array(dbfeatindex['nfeat'])
        topidxtable = np.array(dbfeatindex['topidx'])

        nametable = np.array([os.path.splitext(nm)[0] for nm in nametable])

        indextable = []

        for idx, n in enumerate(featntable):

            indextable += n*[idx]

        indextable = np.array(indextable, dtype=np.int32)

        return nametable, indextable, featntable, topidxtable

    except Exception as e:
        sys.stderr.write("Could not read feature indexing file!\n")
        e_type, e_val, e_tb = sys.exc_info()
        traceback.print_exception(e_type, e_val, e_tb)
        return None, None, None, None