#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import sys
import glob
import argparse
import time

import cv2
from sklearn.preprocessing import normalize
import numpy as np

from libretrieval.ranking.score import *
from libretrieval.features.io import load_features
from libretrieval.utility import safe_create_dir
from libretrieval.search.query import flann_search
#from libretrieval.utility import load_feature_indexing

from search_index import get_index_path
from create_ranks import invert_index

verbose = True

completedir = lambda d: d if d[-1] == "/" else d + "/"


def invert_index(db_namelist):

    invidx = np.zeros(sum(db_namelist['nfeat']), dtype=np.uint32)

    sidx = 0
    for i, n in enumerate(db_namelist['nfeat']):
        eidx = sidx + n
        invidx[sidx:eidx] = i
        sidx = eidx

    return invidx


def search_i(qname, retcfg, outdir):

    ts = dict(find=0,
              loadf=0,
              loadi=0,
              search=0,
              rank=0,
              elapsed=0)

    ts['elapsed'] = time.perf_counter()

    outdir = completedir(outdir)
    safe_create_dir(outdir)

    # Loads the listing of DB features and generates inverted index
    db_namelist = np.loadtxt(retcfg['path']['dblist'], dtype=dict(names=('name', 'nfeat'), formats=('U100', np.int32)))
    dbsize = db_namelist['name'].shape[0]
    invidx = invert_index(db_namelist)

    ### Features config ###
    norm = retcfg.get('feature', 'norm', fallback=None)
    ### Ranking config ###
    score = retcfg['rank']['score_type']
    limit = retcfg.getint('rank', 'limit')
    exfirst = retcfg.getboolean('rank', 'exclude_first')
    exzero = retcfg.getboolean('rank', 'exclude_zero')
    ### Search config ###
    search_type = retcfg['search']['search_type']
    knn = retcfg.getint('search', 'knn')
    rfactor = retcfg.getfloat('search', 'radius_factor')

    # Load and normalize db features
    ts['loadf'] = time.perf_counter()
    db_features = load_features(retcfg['path']['dbfeature'])
    if norm:
        db_features = normalize(db_features, norm)
    ts['loadf'] = time.perf_counter() - ts['loadf']

    # Find query features
    ts['find'] = time.perf_counter()
    try:
        qi = np.argwhere(db_namelist['name'] == qname).reshape(-1)[0]   # Position
        print("$$$ ", qi)
    except IndexError as ie:
        print(". Query image <", qname, "> not found.")
        print("Exiting...")
        return

    nfeat = db_namelist['nfeat'][qi]                                   # Number of Features
    q_features = db_features[qi:qi+nfeat]
    ts['find'] = time.perf_counter() - ts['find']

    # Loading index
    ts['1oadi'] = time.perf_counter()
    fidx = cv2.flann_Index()
    ifile = get_index_path(retcfg['path']['outdir'], retcfg['DEFAULT']['expname'])
    fidx.load(db_features, ifile)
    ts['loadi'] = time.perf_counter() - ts['loadi']

    print(". Searching: ", qname)
    print("   |_ Position: ", qi)
    print("   |_ # Features: ", nfeat, end="\n\n")

    # Searching
    ts['search'] = time.perf_counter()
    votes, dists, _ = flann_search(q_features, fidx, stype=search_type, k=knn, f=rfactor, flib="cv")
    ts['search'] = time.perf_counter() - ts['search']

    # Ranking
    ts['rank'] = time.perf_counter()
    votescores = np.zeros(dbsize, dtype=np.float32)
    distscores = np.zeros(dbsize, dtype=np.float32)
    namearray = db_namelist['name']

    for i in range(nfeat):

        # v has the index of the feature that got the vote. invidx[v] is the index of the
        # collection image the feature v is from. Thus, 1 is summed in the number of votes
        # for the feature, and the distance got is summed as well (to be divided later)
        for v, d in zip(votes[i, :], dists[i, :]):
            dbi = invidx[v]
            votescores[dbi] += 1
            distscores[dbi] += d

    distscores = distscores / votescores

    aux = np.logical_not(np.logical_or(np.isnan(distscores), np.isinf(distscores)))
    votescores = votescores[aux]
    distscores = distscores[aux]
    namearray = namearray[aux]

    normvotes, _ = normalize_scores(votescores.reshape(-1, 1), minmax_range=(1, 2), cvt_sim=False)
    normdists, _ = normalize_scores(distscores.reshape(-1, 1), minmax_range=(1, 2), cvt_sim=True)

    normvotes = normvotes.reshape(-1)
    normdists = normdists.reshape(-1)

    # Excludes first result. Used to exclude the query image, in case it is present in the database.
    if exfirst:
        mv = np.max(votescores)
        aux = votescores != mv
        votescores = votescores[aux]
        distscores = distscores[aux]
        namearray = namearray[aux]

    # Excludes any results with 0 votes.
    if exzero:
        aux = votescores != 0
        votescores = votescores[aux]
        distscores = distscores[aux]
        namearray = namearray[aux]

    aux = zip(namearray, votescores, normvotes, distscores, normdists)

    dt = dict(names=('name', 'votes', 'normv', 'dists', 'normd'),
              formats=('U100', np.float32, np.float32, np.float32, np.float32))

    rank = np.array([a for a in aux], dtype=dt)

    # norm distances are used to sort, instead of pure mean distances, as they are normalized to a similarity,
    # thus being being in accord to voting, which is also a similarity.
    if score == "vote":
        rank.sort(order=('votes', 'normd', 'name'))
        rank = rank[::-1]  # Just inverting the sort order to highest scores

    elif score == "distance":
        rank.sort(order=('normd', 'votes', 'name'))
        rank = rank[::-1]  # Just inverting the sort order to highest scores

    if limit < 0:
        limit = rank.shape[0]
    ts['rank'] = time.perf_counter() - ts['rank']

    rankfpath = "{0:s}{1:s}.rk".format(outdir, qname)
    np.savetxt(rankfpath, rank[0:limit], fmt="%-50s %10.5f %10.5f %10.5f %10.5f")

    ts['elapsed'] = time.perf_counter() - ts['elapsed']

    ts = dict(find=0,
              loadf=0,
              loadi=0,
              search=0,
              rank=0,
              elapsed=0)

    print("      -> Loading DB features: {0:0.3f}s".format(ts['loadf']))
    print("      -> Finding feature: {0:0.3f}s".format(ts['find']))
    print("      -> Loading index: {0:0.3f}s".format(ts['loadi']))
    print("      -> Searching: {0:0.3f}s".format(ts['search']))
    print("      -> Building Rank: {0:0.3f}s".format(ts['rank']))
    print("      -> Total: {0:0.3f}s".format(ts['elapsed']), end="\n\n")






