#!/usr/bin/env python
#-*- coding: utf-8 -*-


import argparse
import glob

import numpy as np

from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.neighbors import NearestNeighbors

from libretrieval.utility import safe_create_dir, cfgloader
from libretrieval.features.io import load_features

import nmslib

from tqdm import tqdm

from create_ranks import invert_index

import ipdb as pdb

batch_size = 500

def get_order(matchscore, distscore):

    normdistscore = normalize(normdist)
    normdistscore = np.max(normdist) - normdist

    comb = matchscore + normdistscore

    return np.argsort(comb)[::-1]


def create_and_search_db_local(retcfg, jobs):

    norm = retcfg.get('feature', 'norm', fallback=None)

    print(" -- loading DB features from: {0:s}".format(retcfg['path']['dbfeature']))
    db_features = load_features(retcfg['path']['dbfeature'])
    db_namelist = np.loadtxt(retcfg['path']['dblist'], dtype=dict(names=('qname', 'nfeat'), formats=('U100', np.int32)))
    ns = db_namelist.shape[0]
    idarray = np.arange(ns).astype(np.int32)

    invidx = invert_index(db_namelist)

    score = retcfg.get('rank', 'score_type', fallback='vote')

    if norm:
        db_features = normalize(db_features, norm)

    outdir = retcfg['path']['outdir']
    safe_create_dir(outdir)

    index_type = retcfg['index']['index_type']
    dist_type = retcfg['index']['dist_type']

    knn = retcfg.getint('search', 'knn_db', fallback=10)
    nmatches = retcfg.getint('rank', 'limit', fallback=100)

    print(" -- Creating <{0:s}> NN index".format(index_type))
    print("     -> KNN: {0:d}".format(knn))
    print("     -> Metric: {0:s}\n".format(dist_type))

    nnidx = nmslib.init(method=index_type, space=dist_type)
    nnidx.addDataPointBatch(db_features)
    nnidx.createIndex({'post': 2}, print_progress=True)
    nnidx.setQueryTimeParams({'efSearch': knn})

    indices = np.zeros((db_namelist.shape[0], nmatches), dtype=np.int32) - 1
    scores = np.zeros((db_namelist.shape[0], nmatches), dtype=np.float64) - 1

    s = 0

    np.seterr(divide='ignore')
    for i in tqdm(range(10), ncols=100, desc='Sample #', total=10):

        name, nf = db_namelist[i]
        e = s + nf

        batch_q_features = db_features[s:e]
        neighbours = np.array(nnidx.knnQueryBatch(batch_q_features, k=knn, num_threads=jobs))
        neighbours = list(zip(*neighbours))

        pdb.set_trace()

        indices_ = np.array(neighbours[0]).reshape(-1).astype(np.int32)
        dists_ = np.array(neighbours[1]).reshape(-1)

        matchscore = np.bincount(invidx[indices_], minlength=ns)
        distscores = np.bincount(invidx[indices_], weights=dists_, minlength=ns)/matchscore

        aux = np.logical_not(np.logical_or(np.isnan(distscores), np.isinf(distscores)))
        matchscore = matchscore[aux]
        distscores = distscores[aux]
        idarray_ = idarray[aux]

        if score == "vote":
            finalscore = matchscore
        elif score == "distance":
            finalscore = distscores
        elif score == "combine":
            # Norm L2 and convert to similarity
            distscores_n = normalize(distscores.reshape(1, -1)).reshape(-1)
            distscores_n = np.max(distscores_n) - distscores_n

            finalscore = matchscore + distscores_n

        order = np.argsort(finalscore)
        if score == "vote" or score == "combine":
            order = order[::-1]

        idarray_ = idarray_[order]
        finalscore = finalscore[order]

        indices[i, :idarray_.size] = idarray_
        scores[i, :idarray_.size] = finalscore

        s = e



    assert np.argwhere(indices == -1).size == 0, "Indices on positions {0:s} have not been updated " \
                                                 "correctly".format(np.argwhere(indices == -1).tostring())

    assert np.argwhere(scores == -1).size == 0, "Indices on positions {0:s} have not been updated " \
                                                 "correctly".format(np.argwhere(indices == -1).tostring())


    outfile = "{0:s}{1:s}_db_matches.npy".format(outdir, retcfg.get('DEFAULT', 'expname'))
    np.save(outfile, indices)

    outfile = "{0:s}{1:s}_db_scores.npy".format(outdir, retcfg.get('DEFAULT', 'expname'))
    np.save(outfile, scores)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("cfgfile", help="Retrieval configuration file", type=str)
    parser.add_argument("-j", "--njobs", help="Number of concurrent jobs for searching", type=int, default=1)
    args = parser.parse_args()

    print(" -- Loading cfg")
    retcfg = cfgloader(args.cfgfile)

    create_and_search_db_local(retcfg, args.njobs)
