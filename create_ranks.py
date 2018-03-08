#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import sys
import glob
import numpy as np
import argparse
import time

from libretrieval.ranking.score import *
#from libretrieval.utility import load_feature_indexing

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

def create_rank_files(retcfg):

    outdir = completedir(retcfg['path']['outdir']) + "queryfiles/"

    db_namelist = np.loadtxt(retcfg['path']['dblist'], dtype=dict(names=('name', 'nfeat'), formats=('U100', np.int32)))
    dbsize = db_namelist['name'].shape[0]

    matchflist = glob.glob(outdir + "*.matches*")
    distflist = glob.glob(outdir + "*.dist*")

    assert len(matchflist) == len(distflist), "Inconsistent number of match and distance files."

    matchflist.sort()
    distflist.sort()

    score = retcfg['rank']['score_type']
    limit = retcfg.getint('rank', 'limit')
    exfirst = retcfg.getboolean('rank', 'exclude_first')
    exzero = retcfg.getboolean('rank', 'exclude_zero')

    invidx = invert_index(db_namelist)

    for matchfpath, distfpath in zip(matchflist, distflist):

        ts = time.perf_counter()
        basename = os.path.basename(matchfpath).rsplit('.', 2)[0]

        rankfpath = "{0:s}{1:s}.rk".format(outdir, basename)
        votes = np.load(matchfpath)
        dists = np.load(distfpath)

        print(matchfpath)
        print(distfpath)
        print(rankfpath)

        assert votes.shape == dists.shape, "Inconsistent shape between votes and distance array"
        nqfeat = votes.shape[0]

        votescores = np.zeros(dbsize, dtype=np.float32)
        distscores = np.zeros(dbsize, dtype=np.float32)
        namearray = db_namelist['name']

        for i in range(nqfeat):

            # v has the index of the feature that got the vote. invidx[v] is the index of the
            # collection image the feature v is from. Thus, 1 is summed in the number of votes
            # for the feature, and the distance got is summed as well (to be divided later)
            for v, d in zip(votes[i, :], dists[i, :]):
                dbi = invidx[v]
                votescores[dbi] += 1
                distscores[dbi] += d

        distscores = distscores/votescores

        aux = np.logical_not(np.logical_or(np.isnan(distscores), np.isinf(distscores)))
        votescores = votescores[aux]
        distscores = distscores[aux]
        namearray = namearray[aux]

        normvotes, _ = normalize_scores(votescores, cvt_sim=False, min_val=0)
        normdists, _ = normalize_scores(distscores, cvt_sim=True, min_val=0)

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

        print("shapes - ", namearray.shape, votescores.shape, normvotes.shape, distscores.shape, normdists.shape)

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

        if limit < 0:  limit = rank.shape[0]
        te = time.perf_counter()
        print("{0:0.4f}s".format(te-ts), end="\n---\n")

        np.savetxt(rankfpath, rank[0:limit], fmt="%-50s %10.5f %10.5f %10.5f %10.5f")
