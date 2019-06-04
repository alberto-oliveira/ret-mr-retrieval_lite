#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import glob

import numpy as np

from tqdm import tqdm


def load_features(fpath):

    if os.path.isfile(fpath):
        features = np.load(fpath)

        return features

    elif os.path.isdir(fpath):

        filelist = glob.glob(fpath + "/*.npy")
        assert len(filelist) > 0, "No .npy files found on input directory"

        filelist.sort()

        features = []
        for f in tqdm(filelist, ncols=100, desc='Feat. File #', total=len(filelist)):

            features.append(np.load(f))

        return np.concatenate(features)

    else:
        raise OSError("<{0:s}> is neither file nor directory".format(fpath))
