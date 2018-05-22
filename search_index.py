#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import sys
import argparse
import glob
import numpy as np
import cv2
from sklearn.preprocessing import normalize

from libretrieval.utility import safe_create_dir
from libretrieval.search.query import flann_search
from libretrieval.features.io import load_features

def get_index_path(indexdir, prefix):

    aux = glob.glob(indexdir + "{0:s}*".format(prefix))

    if len(aux) == 0:
        raise OSError("Index file with prefix <{0:s}> not found on directory <{1:s}>.".format(prefix, indexdir))
    else:
        return aux[0]

def write_times_file(timesfpath, qnamel, timesl):

    mnt = np.mean(timesl)

    tf = open(timesfpath, 'w')

    for qn, t in zip(qnamel, timesl):
        tf.write("{0:<40s} {1:0.5f}\n".format(qn, t))

    tf.write("-----------------\n{0:<40s} {1:0.5f}".format("mean", mnt))

    tf.close()

def search_index(retcfg):

    q_features = load_features(retcfg['path']['qfeature'])
    q_namelist = np.loadtxt(retcfg['path']['qlist'], dtype=dict(names=('qname', 'nfeat'), formats=('U100', np.int32)))

    assert q_features.shape[0] == np.sum(q_namelist['nfeat']), "Inconsistent number of features sum and size of" \
                                                               "query features array"

    norm = retcfg.get('feature', 'norm', fallback=None)

    db_features = load_features(retcfg['path']['dbfeature'])
    if norm:
        db_features = normalize(db_features, norm)

    fidx = cv2.flann_Index()
    ifile = get_index_path(retcfg['path']['outdir'], retcfg['DEFAULT']['expname'])
    fidx.load(db_features, ifile)

    outdir = retcfg['path']['outdir'] + "queryfiles/"
    safe_create_dir(outdir)

    search_type = retcfg['search']['search_type']
    knn = retcfg.getint('search', 'knn')
    rfactor = retcfg.getfloat('search', 'radius_factor')

    sidx = 0
    for qname, n in q_namelist:
        qfeat = q_features[sidx:sidx+n]

        if norm:
            qfeat = normalize(qfeat, norm)

        matchfpath = "{0:s}{1:s}.matches".format(outdir, qname)
        distfpath = "{0:s}{1:s}.dist".format(outdir, qname)

        votes, dists, _ = flann_search(qfeat, fidx, stype=search_type, k=knn, f=rfactor, flib="cv")

        print(qname, "-> ", sidx, ":", sidx+n)
        print(votes.shape)
        print(dists.shape, end="\n---\n")
        np.save(matchfpath + ".npy", votes)
        np.save(distfpath + ".npy", dists)
        sidx += n



