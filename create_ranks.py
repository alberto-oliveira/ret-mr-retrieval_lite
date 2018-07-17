#!/usr/bin/env python
#-*- coding: utf-8 -*-

import glob
import argparse
import time

from libretrieval.ranking.score import *
from libretrieval.utility import cfgloader

completedir = lambda d: d if d[-1] == "/" else d + "/"


def invert_index(db_namelist):

    invidx = np.zeros(sum(db_namelist['nfeat']), dtype=np.uint32)

    sidx = 0
    for i, n in enumerate(db_namelist['nfeat']):
        eidx = sidx + n
        invidx[sidx:eidx] = i
        sidx = eidx

    return invidx

def create_rank_files(retcfg, verbose=True):

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

        matchscore= np.zeros(dbsize, dtype=np.float32)
        distscores = np.zeros(dbsize, dtype=np.float32)
        namearray = db_namelist['name']

        for i in range(nqfeat):

            # v has the index of the feature that got the vote. invidx[v] is the index of the
            # collection image the feature v is from. Thus, 1 is summed in the number of votes
            # for the feature, and the distance got is summed as well (to be divided later)
            for v, d in zip(votes[i, :], dists[i, :]):
                dbi = invidx[v]
                matchscore[dbi] += 1
                distscores[dbi] += d

        if score == "vote":
            finalscore = matchscore
        elif score == "distance":
            finalscore = distscores/matchscore

        aux = np.logical_not(np.logical_or(np.isnan(finalscore), np.isinf(finalscore)))
        finalscore = finalscore[aux]
        namearray = namearray[aux]

        aux = zip(namearray, finalscore)

        dt = dict(names=('name', 'score'),
                  formats=('U100', np.float64))

        rank = np.array([a for a in aux], dtype=dt)
        rank.sort(order=('score', 'name'))

        if score == "vote":
            rank = rank[::-1]

        if limit < 0:
            limit = rank.shape[0]
        te = time.perf_counter()
        print("{0:0.4f}s".format(te-ts), end="\n---\n")

        np.savetxt(rankfpath, rank[0:limit], fmt="%-50s %10.5f %10.5f %10.5f %10.5f")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("cfgfile", help="Retrieval configuration file", type=str)
    args = parser.parse_args()

    retcfg = cfgloader(args.cfgfile)

    create_rank_files(retcfg)
