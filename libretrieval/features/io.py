#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import glob

import numpy as np


def load_features(fpath):

    if os.path.isfile(fpath):
        features = np.load(fpath)
    elif os.path.isdir(fpath):

        filelist = glob.glob(fpath + "/*.npy")
        assert len(filelist) > 0, "No .npy files found on input directory"

        filelist.sort()

        featlist = [np.load(f) for f in filelist]
        features = np.vstack(featlist)

    else:
        raise OSError("<{0:s}> is neither file nor directory".format(fpath))

    return features
