#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys
import os
import argparse
import cv2
import glob
import numpy as np
import traceback
import time

from libretrieval.io import *
from libretrieval.search.pfidx import load_pf_flann_index
from libretrieval.search.query import flann_search
from libretrieval.ranking.score import *
from libretrieval.features.keypoints import keypoints_from_array
from libretrieval.utility import safe_create_dir
from libretrieval.consistency.spatial import *
from libretrieval.draw import *

verbose = False

mrknum = 5000

def save_rank(rankfpath, namelist, votescores, normvotes, distscores, normdists):

    aux = zip(namelist[0:mrknum], votescores[0:mrknum], normvotes[0:mrknum], distscores[0:mrknum],
              normdists[0:mrknum])
    dt = dict(names=('qname', 'votes', 'normv', 'dists', 'normd'), formats=('S100', 'f32', 'f32', 'f32', 'f32'))
    rank = np.array(aux, dtype=dt)

    # norm distances are used to sort, instead of pure mean distances, as they are normalized to a similarity,
    # thus being being in accord to voting, which is also a similarity.

    rank.sort(order=('votes', 'normd', 'qname'))
    rank = rank[::-1]  # Just inverting the sort order to highest scores

    np.savetxt(rankfpath, rank, fmt="%-50s %10.5f %10.5f %10.5f %10.5f")

def draw_keypoints_img(baseimg, outimgpath, keyp_array):

    keyps = keypoints_from_array(keyp_array)

    imgkeyp = cv2.drawKeypoints(baseimg, keyps, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    cv2.imwrite(outimgpath, imgkeyp)

    return



def count_scores(indextable, votes, distances, multi=False):

    maxi = indextable[-1]

    votetable = [0.0]*(maxi+1)
    distsum = [0.0]*(maxi+1)
    matchedfeat = {}

    # Iterates over the votes (matches).
    r, c = votes.shape
    for ri in xrange(r):
        voted = []
        vrow = votes[ri, :]

        for ci in xrange(c):
            voteidx = vrow[ci]

            # indextable indexes a feature number to a DB image.
            imidx = indextable[voteidx]

            if imidx not in voted:

                distval = distances[ri, ci]

                votetable[imidx] += 1.0
                distsum[imidx] += distval
                if imidx not in matchedfeat:
                    matchedfeat[imidx] = [[ri, voteidx]]
                else:
                    matchedfeat[imidx].append([ri, voteidx])

                # Do not allow two votes to the same image per query descriptor.
                if not multi:
                    voted.append(imidx)

    try:

        np.seterr(divide='ignore', invalid='ignore')

        votescores = np.array(votetable, dtype=np.float64)

        distscores = np.array(distsum, dtype=np.float64) / votescores
        distscores[votescores == 0] = np.inf

    except:
        sys.stderr.write("Problem creating score tables!\n")
        e_type, e_val, e_tb = sys.exc_info()
        traceback.print_exception(e_type, e_val, e_tb)
        return None, None


    return votescores, distscores, matchedfeat

def create_index_table(feature_number):

    indextable = []

    for idx, n in enumerate(feature_number):

        indextable += n*[idx]

    return indextable


def get_db_img_features(dbfeatures, dbkeyps, nfeat, topidx):

    botidx = topidx-nfeat

    imgfeat = dbfeatures[botidx:topidx]
    imgkeyp = dbkeyps[botidx:topidx]

    return imgfeat, imgkeyp

def feature_selection_transform(dbfeatdir, dbindexpath, dbimdir, outdir, sel_crit, limit, prefix):

    minv = 1
    topr = 20
    im_limit = int(np.ceil(0.25*limit))

    dbfeatfpath = glob.glob(dbfeatdir + "*.bfv")[0]
    dbkeypfpath = glob.glob(dbfeatdir + "*.bkp")[0]
    dbidxfpath = glob.glob(dbfeatdir + "*.txt")[0]

    outfeatfpath = "{0:s}{1:s}.bfv".format(outdir, prefix)
    outkeypfpath = "{0:s}{1:s}.bkp".format(outdir, prefix)
    outidxfpath = "{0:s}{1:s}_indexing.txt".format(outdir, prefix)
    outrankdir = "{0:s}ranks/".format(outdir)
    safe_create_dir(outrankdir)

    selfeat = []
    selkeyp = []
    selidx = []

    if verbose: print "Reading DB features and keypoints: "
    dbfeatures = read_array_bin_file(dbfeatfpath)
    dbkeypoints = read_array_bin_file(dbkeypfpath)
    if verbose: print "  -> ", dbfeatfpath, " ", dbfeatures.shape
    if verbose: print "  -> ", dbkeypfpath, " ", dbkeypoints.shape, "\n"

    if verbose: print "Reading Indexing file: ",
    if verbose: print "  -> ", dbidxfpath, "\n"
    idxdt = dict(names=('name', 'nfeat', 'topidx'), formats=('S100', 'i32', 'i32'))
    dbfeatindex = np.loadtxt(dbidxfpath, dtype=idxdt)
    nametable = dbfeatindex['name']
    indextable = create_index_table(dbfeatindex['nfeat'])

    if verbose: print "Loading FLANN index: ",
    if verbose: print "  -> ", dbindexpath, "\n"
    flann_index = load_pf_flann_index(dbfeatures, dbindexpath)

    if verbose: print " -- Starting feature selection --"
    for dbimname, nf, ti in dbfeatindex:
        ts_fs = time.time()
        dbimpath = glob.glob(dbimdir + dbimname)[0]
        rankfpath = outrankdir + "rank_" + os.path.splitext(dbimname)[0] + ".rk"
        dbim = cv2.imread(dbimpath)

        imindexes = np.arange(nametable.shape[0])

        if verbose: print "  -> Selecting features of: ", dbimname, " ({0:d}, {1:d}) from {2:d}".format(ti-nf, ti, dbfeatures.shape[0])
        if verbose: print "    --> Selection criteria: ", sel_crit
        if verbose: print "    --> Minimum votes: ", minv
        if verbose: print "    --> Rank positions: top ", topr
        qfeat, qkeyp = get_db_img_features(dbfeatures, dbkeypoints, nf, ti)

        votes, dists, tt = flann_search(qfeat, flann_index, stype='knn', k=10, flib="pf")
        if verbose: print "    --> FLANN search time: {0:0.3f}s".format(tt)

        votescores, distscores, matchedfeat = count_scores(indextable, votes, dists)
        if verbose: print "    --> Total number of matched images: ", len(matchedfeat), "\n"

        aux = votescores != 0
        votescores = votescores[aux]
        distscores = distscores[aux]
        resnametable = nametable[aux]
        imindexes = imindexes[aux]

        normvotes, _ = normalize_scores(votescores, cvt_sim=False, min_val=0)
        normdists, _ = normalize_scores(distscores, cvt_sim=True, min_val=0)

        save_rank(rankfpath, resnametable, votescores, normvotes, distscores, normdists)

        mv = np.max(votescores)
        idx_choice = np.logical_and(votescores != mv, votescores >= minv)

        resnametable = resnametable[idx_choice]
        votescores = votescores[idx_choice]
        normdists = normdists[idx_choice]
        imindexes = imindexes[idx_choice]

        aux = zip(votescores, normdists)
        dt = dict(names=('votes', 'normd'), formats=('f32', 'f32'))
        scores = np.array(aux, dtype=dt)

        ridx = scores.argsort(order=('votes', 'normd'))
        ridx = ridx[::-1]

        selected_idx = []
        total_sel = 0
        for pos, i in enumerate(ridx[:topr]):

            imi = imindexes[i]
            if verbose: print "      > Ranked {0:d} (Img {3:d}): {1:s} = {2:f} votes".format(pos, resnametable[i],
                                                                                             votescores[i], imi)

            midx = np.array(matchedfeat[imi])
            #print midx

            qidx = midx[:, 0]
            tidx = midx[:, 1]

            if sel_crit == 'base':

                selected_idx.append(qidx)
                total_sel += qidx.shape[0]

                if verbose: print "        `-> {0:d} features selected".format(qidx.shape[0])

            if sel_crit == 'line':

                qm_keyp = qkeyp[qidx]
                tm_keyp = dbkeypoints[tidx]

                cons, ll, la = line_geom_consistency(qm_keyp[:, 0:2], tm_keyp[:, 0:2], dbim.shape[1], 1.0)

                print "Cons Line: ", cons

                cons_idx = qidx[cons]
                if verbose: print "         * Consistent number: ", cons_idx.shape
                if cons_idx.shape[0] > 0:

                    # If te number of selected features is above the input limit, it applies the image limit. The image
                    # limit is ceil(0.25*input_limit). The features selected with the image limit are those with best
                    # response.
                    if cons_idx.shape[0] > im_limit:
                        qkeyp_sel = qkeyp[cons_idx]                 # Gets the selected keypoints
                        qkeyp_resp = qkeyp_sel[:, 4].reshape(-1)    # Gets the response of the selected keypoints
                        resp_order = qkeyp_resp.argsort()           # Gets the index of the ordered responses
                        cons_idx = cons_idx[resp_order]             # Gets the index of the consis. feat. of highest response
                        cons_idx = cons_idx[:im_limit]              # Gets only the #image_limit consistent features.


                    selected_idx.append(cons_idx)
                    total_sel += cons_idx.shape[0]

                    if verbose: print "        `-> {0:d} features selected".format(cons_idx.shape[0])

                else:
                    if verbose: print "        `-> No features selected"

            if sel_crit == 'transform':

                qm_keyp = qkeyp[qidx]
                tm_keyp = dbkeypoints[tidx]

                _, cons = cv2.findFundamentalMat(qm_keyp[:, 0:2].astype(np.float32),
                                                 tm_keyp[:, 0:2].astype(np.float32),
                                                 method=cv2.FM_RANSAC,
                                                 param1=3,
                                                 param2=0.99)

                cons = cons.reshape(-1).astype('bool')

                cons_idx = qidx[cons]
                if verbose: print "         * Consistent number: ", cons_idx.shape
                if cons_idx.shape[0] > 0:

                    # If te number of selected features is above the input limit, it applies the image limit. The image
                    # limit is ceil(0.25*input_limit). The features selected with the image limit are those with best
                    # response.
                    if cons_idx.shape[0] > im_limit:
                        qkeyp_sel = qkeyp[cons_idx]                 # Gets the selected keypoints
                        qkeyp_resp = qkeyp_sel[:, 4].reshape(-1)    # Gets the response of the selected keypoints
                        resp_order = qkeyp_resp.argsort()           # Gets the index of the ordered responses
                        cons_idx = cons_idx[resp_order]             # Gets the index of the consis. feat. of highest response
                        cons_idx = cons_idx[:im_limit]              # Gets only the #image_limit consistent features.


                    selected_idx.append(cons_idx)
                    total_sel += cons_idx.shape[0]

                    if verbose: print "        `-> {0:d} features selected".format(cons_idx.shape[0])

                else:
                    if verbose: print "        `-> No features selected"


            if limit != -1 and total_sel > limit:
                break

        if selected_idx != []:
            selected_idx = np.unique(np.hstack(selected_idx))

            if verbose: print "    --> Selected features:  <{0:d}> from <{1:d}>".format(selected_idx.shape[0], qfeat.shape[0])

            if len(selidx) == 0:
                selidx.append((dbimname, selected_idx.shape[0], selected_idx.shape[0]))
            else:
                prev_top = selidx[-1][2]
                selidx.append((dbimname, selected_idx.shape[0], prev_top + selected_idx.shape[0]))

            selfeat.append(qfeat[selected_idx])
            selkeyp.append(qkeyp[selected_idx])

        else:
            print "    --> No features selected frm image"

        #print "    --> Drawing output images"
        #outimdir = "{0:s}{1:s}".format(outdrawdir, os.path.splitext(dbimname)[0])
        #safe_create_dir(outimdir)

        #draw_keypoints_img(dbim, outimdir + "/original_keypoints_" + dbimname, qkeyp)
        #draw_keypoints_img(dbim, outimdir + "/{0:s}_selected_keypoints_{1:s}".format(sel_crit, dbimname), qkeyp[selected_idx])

        te_fs = time.time()
        if verbose: print " - Done ({0:0.3f}s) - \n\n".format(te_fs - ts_fs)

    selected_db_features = np.vstack(selfeat)
    selected_db_keypoints = np.vstack(selkeyp)
    selected_feat_indexing = np.array(selidx, dtype=idxdt)

    if verbose: print "final feat shape: ", selected_db_features.shape
    if verbose: print "final keyp shape: ", selected_db_keypoints.shape
    if verbose: print "final indexing shape: ", selected_feat_indexing.shape


    write_array_bin_file(outfeatfpath, selected_db_features)
    write_array_bin_file(outkeypfpath, selected_db_keypoints)
    np.savetxt(outidxfpath, selected_feat_indexing, fmt="%-50s %d %d")






if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("dbfeatdir",
                        help="Directory containing DB feature (.bfv) and keypoint (.bkp) files",
                        type=str)

    parser.add_argument("dbindexpath",
                        help="Path to the FLANN index file.")

    parser.add_argument("dbimdir",
                        help="Directory with db images")

    parser.add_argument("outdir",
                        help="Output directory for feature and keypoint files, after selection",
                        type=str)

    parser.add_argument("--sel_crit", "-s",
                        help="Output directory for feature and keypoint files, after selection",
                        choices=['base', 'line', 'transform'],
                        default='none',
                        type=str)

    parser.add_argument("--limit", "-l",
                        help="Limit the features to be chosen. If -1, takes all of the select. Default is -1",
                        type=int,
                        default=-1)

    parser.add_argument("--prefix", "-p",
                        help="Prefix to name output files. Default is \'selected_features\' ",
                        type=str,
                        default="selected_features")

    parser.add_argument("--verbose", "-v",
                        help="Printing of additional info",
                        action="store_true")

    args = parser.parse_args()
    dbfeatdir = args.dbfeatdir
    dbindexpath = args.dbindexpath
    dbimdir = args.dbimdir
    outdir = args.outdir
    sel_crit = args.sel_crit
    limit = args.limit
    prefix = args.prefix
    verbose = args.verbose

    if dbfeatdir[-1] != "/": dbfeatdir += "/"
    if outdir[-1] != "/": outdir += "/"
    if dbimdir[-1] != "/": dbimdir += "/"

    feature_selection_transform(dbfeatdir, dbindexpath, dbimdir, outdir, sel_crit, limit, prefix)