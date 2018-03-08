#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import argparse
import glob
import errno

import numpy as np

from libretrieval.utility import safe_create_dir
from libretrieval.features.io import load_features

from libretrieval.search.index import create_flann_index

def create_index(retcfg):

    index_type = retcfg['index']['index_type']
    dist_type = retcfg['index']['dist_type']
    lib = retcfg['index']['lib']

    safe_create_dir(retcfg['path']['outdir'])

    dbfeatures = load_features(retcfg['path']['dbfeature'])

    outpath = "{0:s}{1:s}_{2:s}_{3:s}.dat".format(retcfg['path']['outdir'], retcfg['DEFAULT']['expname'],
                                                  index_type, dist_type)

    if lib == "cv":
        fidx, params = create_flann_index(dbfeatures, index_type, dist_type)
    else:
        raise ValueError("Unsupported flann lib <{0:s}>".format(lib))

    fidx.save(outpath)

    return
