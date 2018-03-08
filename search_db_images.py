#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""

search_db_images.py

Perform queries using each DB image.

usage: search_db_images.py [-h] [--verbose] dbfeatdir idxpath outdir

positional arguments:
  dbfeatdir      Directory containing: dbfeatures (.bfv), dbkeypoints (.bkp),
                 and indexing file (.txt)
  idxpath        Flann index path. Must be compatible with the features
                 loaded.
  outdir         Output directory for rank files.

optional arguments:
  -h, --help     show this help message and exit
  --verbose, -v  Printing of additional info.

Output: ranked lists of DB images.

"""

import sys
import os
import argparse
import glob

from objret.io import *

from objret.search.pfidx import load_pf_flann_index

from objret.utility import map_config, safe_create_dir

from objret.search.query import flann_search

from objret.ranking.score import *

config_path = './search_config.conf'


def get_features(dbfeatures, dbkeyps, nfeat, topidx):

    botidx = topidx-nfeat

    imgfeat = dbfeatures[botidx:topidx]
    imgkeyp = dbkeyps[botidx:topidx]

    return imgfeat, imgkeyp


def create_index_table(feature_number):

    indextable = []

    for idx, n in enumerate(feature_number):

        indextable += n*[idx]

    return indextable


def search_db_images(dbfeatdir, idxpath, outdir, verbose=False):

    safe_create_dir(outdir)

    configm = map_config(config_path)

    dbfeatfpath = glob.glob(dbfeatdir + "*.bfv")[0]
    dbkeypfpath = glob.glob(dbfeatdir + "*.bkp")[0]
    dbidxfpath = glob.glob(dbfeatdir + "*.txt")[0]

    if verbose: print "Reading DB features and keypoints: "
    dbfeatures = read_array_bin_file(dbfeatfpath)
    dbkeypoints = read_array_bin_file(dbkeypfpath)
    if verbose: print "  -> ", dbfeatfpath, " ", dbfeatures.shape
    if verbose: print "  -> ", dbkeypfpath, " ", dbkeypoints.shape, "\n"

    ndbf = dbfeatures.shape[0]

    if verbose: print "Reading Indexing file: ",
    if verbose: print "  -> ", dbidxfpath, "\n"
    idxdt = dict(names=('name', 'nfeat', 'topidx'), formats=('S100', 'i32', 'i32'))
    dbfeatindex = np.loadtxt(dbidxfpath, dtype=idxdt)
    nametable = np.array([os.path.splitext(nm)[0] for nm in dbfeatindex['name']])
    indextable = create_index_table(dbfeatindex['nfeat'])

    if verbose: print "Loading FLANN index: ", idxpath
    flann_index = load_pf_flann_index(dbfeatures, idxpath)
    if flann_index is None:
        raise ValueError

    if verbose: print "\n"
    for i, qinfo in enumerate(dbfeatindex):

        qname, qnf, qti = qinfo

        if verbose: print "Running query <{0:s}> - feat. ({1:d} to {2:d}) from {3:d}".format(qname, qti-qnf, qti, ndbf)
        qfeatures, qkeypoints = get_features(dbfeatures, dbkeypoints, qnf, qti)

        if verbose: sys.stdout.write("    -> Searching index (knn = {0:d}...".format(int(configm['n_neighbors'])))
        votes, dists, tt = flann_search(qfeatures,
                                        flann_index,
                                        stype=configm['stype'],
                                        k=int(configm['n_neighbors']),
                                        f=float(configm['rfactor']),
                                        flib='pf')
        if verbose: sys.stdout.write("Done! ({0:0.3f})\n".format(tt))

        vtb = [0.0]*len(nametable)
        rknamearray = np.array(nametable)

        if verbose: print "    -> Counting Votes..."
        votescores, distscores = count_scores(indextable, votes, votetable=vtb, distances=dists, multi=False)


        if verbose: print "    -> Ranking Images..."
        aux = votescores != 0
        votescores = votescores[aux]
        distscores = distscores[aux]
        rknamearray = rknamearray[aux]

        normvotes, _ = normalize_scores(votescores, cvt_sim=False, min_val=0)
        normdists, _ = normalize_scores(distscores, cvt_sim=True, min_val=0)

        rkl = int(configm['rklimit'])
        if rkl < 0: rkl = rknamearray.shape[0]

        aux = zip(rknamearray, votescores, normvotes, distscores,
                  normdists)
        dt = dict(names=('qname', 'votes', 'normv', 'dists', 'normd'), formats=('S100', 'f32', 'f32', 'f32', 'f32'))
        rank = np.array(aux, dtype=dt)

        # norm distances are used to sort, instead of pure mean distances, as they are normalized to a similarity,
        # thus being being in accord to voting, which is also a similarity.
        if configm['score'] == "vote":
            rank.sort(order=('votes', 'normd', 'qname'))
            rank = rank[::-1]  # Just inverting the sort order to highest scores

        elif configm['score'] == "distance":
            rank.sort(order=('normd', 'votes', 'qname'))
            rank = rank[::-1]  # Just inverting the sort order to highest scores

        outrankfpath = "{0:s}{1:s}_{2:s}.rk".format(outdir, configm['prefix'], os.path.splitext(qname)[0])
        if verbose: print "    -> Writing rank file: {0:s} -  <{1:s}> type".format(outrankfpath, configm['score'])
        np.savetxt(outrankfpath, rank[0:rkl], fmt="%-50s %10.5f %10.5f %10.5f %10.5f")

        if verbose: print "\n\n"




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("dbfeatdir",
                        help="Directory containing: dbfeatures (.bfv), dbkeypoints (.bkp), and indexing file (.txt)",
                        type=str)

    parser.add_argument("idxpath",
                        help="Flann index path. Must be compatible with the features loaded.",
                        type=str)

    parser.add_argument("outdir",
                        help="Output directory for rank files.",
                        type=str)

    parser.add_argument("--verbose", "-v",
                        help="Printing of additional info.",
                        action="store_true")

    args = parser.parse_args()


    if args.dbfeatdir[-1] == "/":
        dbfeatdir = args.dbfeatdir
    else:
        dbfeatdir = args.dbfeatdir + "/"

    if args.outdir[-1] == "/":
        outdir = args.outdir
    else:
        outdir = args.outdir + "/"

    idxpath = args.idxpath

    search_db_images(dbfeatdir, idxpath, outdir, args.verbose)
