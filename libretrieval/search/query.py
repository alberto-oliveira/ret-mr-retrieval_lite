#/usr/bin/env python
# -*- coding: utf-8 -*-

"""
objret.search.query.py

Functions for querying flann indexes for matching features in a large database.
"""

import sys
import os
import time
import traceback
import cv2
import numpy as np




def compute_radius(queryfeat, flann_index, k=10, f=1.0):
    """ Computes a distance radius based on a subset of query features.

    Uses the 10 first query features to compute a search radius, based on their k nearest neightbors. The radius is
    computed as:

    r = m + f*s

    where m is the mean distance, s is the standard deviation of the distances and f an input factor.

    :param queryfeat: query features;
    :param flann_index: flann index to search for the DB features;
    :param k: number of neighbors considered;
    :param f: multiplying factor for the standard deviation;

    :return: computed radius;
    """


    samplefeat = queryfeat[0:10, :]

    samp_ind, samp_dist = flann_index.knnSearch(samplefeat, k, params=dict(checks=32))

    mean_max_d = np.mean(samp_dist[:, -1])
    std_max_d = np.std(samp_dist[:, -1])

    radius = mean_max_d + f*std_max_d

    return radius


def flann_search(queryfeat, flann_index, stype="knn", k=1, f=1.0, flib="pf"):
    """ Perform FLANN search, specifying which library was used to construct the index.

    Both native FLANN library and OpenCV's interface are available. Because of small differences between both,
    they should be specified by the \'flib\' parameter.

    :param queryfeat: query features;
    :param flann_index: flann index to search for the DB features;
    :param stype: type of search. Either "radius" or "knn", defaulting to knn
    :param k: Number of neighbors;
    :param f: If radius search, multiplying factor of the standard deviation to compute the radius;
    :param flib: Flann library used. Either "cv" (OpenCV) or "pf" (Native FLANN);

    :return: array of indices of the matching DB features, array of distances of the matching DB features,
             total search time.
    """

    if stype == "radius":

        r = compute_radius(queryfeat, flann_index, k, f)

        if flib == "pf":
            ts = time.perf_counter()
            indices, dists = flann_index.nn_radius(queryfeat,
                                                   radius=r,
                                                   params=dict(checks=32, sorted=True, max_neighbors=k))
            te = time.perf_counter()

        elif flib == "cv":
            ts = time.perf_counter()
            indices, dists = flann_index.radiusSearch(queryfeat,
                                                      radius=r,
                                                      maxResuls=k,
                                                      params=dict(checks=32))
            te = time.perf_counter()
        else:
            raise ValueError("Unrecognized search lib parameter!\n")

        tt = te - ts

    elif stype == "knn":

        if flib == "pf":
            ts = time.perf_counter()
            indices, dists = flann_index.nn_index(queryfeat,
                                                  num_neighbors=k,
                                                  params=dict(checks=32))
            te = time.perf_counter()

        elif flib == "cv":
            ts = time.perf_counter()
            indices, dists = flann_index.knnSearch(queryfeat,
                                                   knn=k,
                                                   params=dict(checks=32))
            te = time.perf_counter()
        else:
            raise ValueError("Unrecognized search lib parameter!\n")

        tt = te - ts

    else:
        raise ValueError("Unrecognized type of search!\n")

    return indices, dists, tt
