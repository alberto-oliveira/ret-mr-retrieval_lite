#!/usr/bin/env python
#-*- coding: utf-8 -*-


import argparse
import glob

import numpy as np

from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.neighbors import NearestNeighbors

from libretrieval.utility import safe_create_dir, cfgloader
from libretrieval.features.io import load_features

from tqdm import tqdm

from create_ranks import invert_index

#import ipdb as pdb

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
    db_namelist = np.loadtxt(retcfg['path']['qlist'], dtype=dict(names=('qname', 'nfeat'), formats=('U100', np.int32)))
    ns = db_namelist.shape[0]

    invidx = invert_index(db_namelist)

    if norm:
        db_features = normalize(db_features, norm)

    outdir = retcfg['path']['outdir']
    safe_create_dir(outdir)

    index_type = retcfg['index']['index_type']
    dist_type = retcfg['index']['dist_type']

    knn = retcfg.getint('search', 'knn_db', fallback=1000)

    print(" -- Creating <{0:s}> NN index".format(index_type))
    print("     -> KNN: {0:d}".format(knn))
    print("     -> Metric: {0:s}\n".format(dist_type))

    mp = dict()
    if dist_type == 'seuclidean':
        mp['V'] = np.var(db_features, axis=0, dtype=np.float64)

    nnidx = NearestNeighbors(n_neighbors=knn, algorithm=index_type, metric=dist_type, n_jobs=jobs, metric_params=mp)
    nnidx.fit(db_features)

    indices = np.zeros((db_namelist.shape[0], knn), dtype=np.int32) - 1
    scores = np.zeros((db_namelist.shape[0], knn), dtype=np.float64) - 1

    s = 0

    for i in tqdm(range(ns), ncols=100, desc='Sample #', total=ns):

        name, nf = db_namelist[i]
        e = s + nf

        batch_q_features = db_features[s:e]
        dist_, indices_ = nnidx.kneighbors(batch_q_features, return_distance=True)

        idx_ = indices_.reshape(-1)
        dis_ = dist_.reshape(-1)

        matchscore = np.bincount(invidx[idx_], minlegth=ns)
        distscore = np.bincount(invidx[idx_], weights=dis_, minlegth=ns)/matchscore

        np.nan_to_num(distscore)

        order = get_order(matchscore, distscore)

        index = np.range(ns)[order]
        score = matchscore[order]

        index = index[score != 0]
        score = score[score != 0]

        indices[i, 0:order.]

        s = e



    assert np.argwhere(indices == -1).size == 0, "Indices on positions {0:s} have not been updated " \
                                                 "correctly".format(np.argwhere(indices == -1).tostring())

    assert np.argwhere(distances == -1).size == 0, "Indices on positions {0:s} have not been updated " \
                                                 "correctly".format(np.argwhere(indices == -1).tostring())




    outfile = "{0:s}{1:s}_db_matches.npy".format(outdir, retcfg.get('DEFAULT', 'expname'))
    np.save(outfile, indices)


    outfile = "{0:s}{1:s}_db_scores.npy".format(outdir, retcfg.get('DEFAULT', 'expname'))
    np.save(outfile, distances)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("cfgfile", help="Retrieval configuration file", type=str)
    parser.add_argument("-j", "--njobs", help="Number of concurrent jobs for searching", type=int, default=1)
    args = parser.parse_args()

    print(" -- Loading cfg")
    retcfg = cfgloader(args.cfgfile)

    create_and_search_db_local(retcfg, args.njobs)
