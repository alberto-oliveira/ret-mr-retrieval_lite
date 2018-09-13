#/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import traceback

import cv2
import cv2.flann
from sklearn.preprocessing import normalize
import numpy as np
#import pyflann

itype_map = {"LINEAR": 0,
             "KDFOREST": 1,
             "KMEANS": 2,
             "COMPOSITE": 3,
             "KDFOREST_SINGLE": 4,
             "HCAL": 5,
             "LSH": 6,
             "SAVED": 254,
             "AUTO": 255}

dtype_map = {"EUCLIDEAN": 1,
             "L2": 1,
             "MANHATTAN": 2,
             "L1": 2,
             "CHISQUARE": 7,
             "CSQ": 7,
             "KULLBACK-LEIBLER": 8,
             "KBL": 8,
             "HAMMING": 9,
             "HMD": 9}

def create_flann_index(features, itype, dtype, norm=None):
    """ Creates a flann index using OpenCV's FLANN interface

    :param features: array of features to build the index for;
    :param itype: type of index {"LINEAR", "KDFOREST", "KMEANS", "COMPOSITE", "KDFOREST_SINGLE", "HCAL", "LSH", "SAVED",
             "AUTO"};
    :param dtype: type of distance {"EUCLIDEAN" or "L2", "MANHATTAN" or "L1", "CHISQUARE" or "CSQ",
                  "KULLBACK-LEIBLER" or "KBL", "HAMMING" or "HMD"};

    :return: FLANN index.
    """

    if norm:
        features = normalize(features, norm)

    if itype == "LSH":

        idx_params = dict(algorithm=itype_map[itype],
                          table_number=20,
                          key_size=15,
                          multi_probe_level=2)

        flann_index = cv2.flann_Index(features, idx_params, distType=dtype_map[dtype])


    elif itype == "KDFOREST":

        idx_params = dict(algorithm=itype_map[itype],
                          trees=10)

        flann_index = cv2.flann_Index(features, idx_params, distType=dtype_map[dtype])


    elif itype == "KMEANS":

        idx_params = dict(algorithm=itype_map[itype],
                          branching=32,
                          iterations=11,
                          flann_centers_init=0,
                          cb_index=0.2)

        flann_index = cv2.flann_Index(features, idx_params, distType=dtype_map[dtype])


    elif itype == "AUTO":

        idx_params = dict(algorithm=itype_map[itype],
                          target_precision=0.5,
                          build_weight=0.00001,
                          memory_weight=0,
                          sample_fraction=1)

        flann_index = cv2.flann_Index(features, idx_params)


    elif itype == "LINEAR":

        idx_params = dict(algorithm=itype_map[itype])

        flann_index = cv2.flann_Index(features, idx_params, distType=dtype_map[dtype])

    else:
        raise ValueError("Unsupported type of index <{0:s}>".format(itype))

    return flann_index, idx_params