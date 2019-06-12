#!/usr/bin/env python
#-*- coding: utf-8 -*-

import glob
import argparse
import time

from libretrieval.ranking.score import *
from libretrieval.utility import cfgloader

from sklearn.preprocessing import normalize

from tqdm import tqdm

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

    nq = len(matchflist)

    score = retcfg['rank']['score_type']
    limit = retcfg.getint('rank', 'limit')

    invidx = invert_index(db_namelist)

    for i in tqdm(range(nq), ncols=100, desc='Queryfile', total=nq):

        matchfpath = matchflist[i]
        distfpath = distflist[i]

        basename = os.path.basename(matchfpath).rsplit('.', 2)[0]

        rankfpath = "{0:s}{1:s}.rk".format(outdir, basename)
        indices = np.load(matchfpath).astype(np.int32)
        dists = np.load(distfpath)

        assert indices.shape == dists.shape, "Inconsistent shape between indices and distance array"

        namearray = db_namelist['name']

        indices_ = indices.reshape(-1)
        dists_ = dists.reshape(-1)

        matchscore = np.bincount(invidx[indices_], minlength=dbsize)
        distscores = np.bincount(invidx[indices_], weights=dists_, minlength=dbsize)/matchscore

        aux = np.logical_not(np.logical_or(np.isnan(distscores), np.isinf(distscores)))
        matchscore = matchscore[aux]
        distscores = distscores[aux]
        namearray = namearray[aux]

        if score == "vote":
            finalscore = matchscore
        elif score == "distance":
            finalscore = distscores
        elif score == "combine":
            # Norm L2 and convert to similarity
            distscores_n = normalize(distscores.reshape(1, -1)).reshape(-1)
            distscores_n = np.max(distscores_n) - distscores_n

            finalscore = matchscore + distscores_n

        aux = zip(namearray, finalscore)

        dt = dict(names=('name', 'score'),
                  formats=('U100', np.float64))

        rank = np.array([a for a in aux], dtype=dt)
        rank.sort(order=('score', 'name'))

        if score == "vote" or score == "combine":
            rank = rank[::-1]
            rank = rank[rank['score'] != 0]

        if limit < 0:
            limit = rank.shape[0]

        np.savetxt(rankfpath, rank[0:limit], fmt="%-50s %10.5f")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("cfgfile", help="Retrieval configuration file", type=str)
    args = parser.parse_args()

    retcfg = cfgloader(args.cfgfile)

    create_rank_files(retcfg)
