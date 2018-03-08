#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
objret.ranking.score.py

Implements ranking and scoring functions.

"""

import sys
import os
import traceback
import numpy as np
import cv2


def count_scores(indextable, votes, votetable=[], distances=[], multi=False):
    """ Count vote and distance scores.

    Given an indextable, mapping DB features to their source images, and a NxK vote array (N being the number
    of features of the query image and K being the number of neighbors searched), counts the number of votes for
    each image of the DB. An optional NxK distance array can be given as input to compute the average distance
    for the votes of each image.


    :param indextable: table indexing DB features to their source DB images. indextable[i] = X means that feature i
                       belongs to the X-th image;
    :param votes: NxK array with the KNN of the N query features;
    :param votetable: Pre-initialized votetable. If empty, it is initialized inside the function. Array counting the
                      number of votes each image has. votetable[i] = k means that the i-th image has k votes;
    :param distances: NxK array with the distances for the KNN of the N query features;
    :param multi: (optional) Flag indicating if multi-votes can occurs. A multi vote means that a query feature can
                   vote more than once for the same DB image. Defaul is False;
    :return: array counting the number of votes of each DB image, array storing the average vote distance of each DB
             image;
    """

    maxi = indextable[-1]
    if not votetable:
        votetable = [0.0]*(maxi+1)

    distsum = [0.0]*(maxi+1)

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

                # Try to get the distance of the corresponding vote.
                try:
                    distval = distances[ri, ci]
                except (NameError, IndexError):
                    distval = 0.0

                votetable[imidx] += 1.0

                distsum[imidx] += distval

                # Do not allow two votes to the same image per query descriptor.
                if not multi:
                    voted.append(imidx)

    try:

        np.seterr(divide='ignore', invalid='ignore')

        votescores = np.array(votetable, dtype=np.float32)

        distscores = np.array(distsum, dtype=np.float32) / votescores
        distscores[votescores == 0] = np.inf

    except:
        sys.stderr.write("Problem creating score tables!\n")
        e_type, e_val, e_tb = sys.exc_info()
        traceback.print_exception(e_type, e_val, e_tb)
        return None, None


    return votescores, distscores


def count_scores2(indextable, votes, tracks, votetable=[], distances=[], multi=False):
    """ Deprecated. Use count_scores.


    :param indextable:
    :param votes:
    :param tracks:
    :param votetable:
    :param distances:
    :param multi:
    :return:
    """

    return [], []

    maxi = indextable[-1]
    if not votetable:
        votetable = [0.0]*(maxi+1)

    distsum = [0.0]*(maxi+1)

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

                # Try to get the distance of the corresponding vote.
                try:
                    distval = distances[ri, ci]
                except (NameError, IndexError):
                    distval = 0.0

                votetable[imidx] += 1.0

                distsum[imidx] += distval

                # Do not allow two votes to the same image per query descriptor.
                if not multi:
                    voted.append(imidx)

                # Count tracks of features
                track = tracks[voteidx]

                tracked = []

                for tindex in track:

                    if tindex == -1: break

                    trimidix = indextable[tindex]

                    if trimidix != imidx and trimidix not in tracked:

                        votetable[trimidix] += 1.0
                        distsum[trimidix] += distval

                        tracked.append(trimidix)

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


    return votescores, distscores



def count_scores_and_matches(indextable,
                             votes,
                             votetable=[],
                             distances=[],
                             dmatchlist=[],
                             exclude_first=False,
                             exclude_zero=False):
    """ Deprecated. Use count_scores.

    :param indextable:
    :param votes:
    :param votetable:
    :param distances:
    :param dmatchlist:
    :param exclude_first:
    :param exclude_zero:
    :return:
    """

    return [], [], []

    maxi = indextable[-1]
    if not votetable:
        votetable = [0.0]*maxi

    if not dmatchlist:
        dmatchlist = [[] for i in xrange(maxi)]

    distsum = [0.0]*maxi

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

                # Try to get the distance of the corresponding vote.
                try:
                    distval = distances[ri, ci]
                except (NameError, IndexError):
                    distval = 0

                votetable[imidx] += 1.0
                distsum[imidx] += distval

                # ri is the index of the query descriptor. voteidx is the index of the db descriptor.
                dmatchlist[imidx].append(cv2.DMatch(_queryIdx=ri, _trainIdx=voteidx, _distance=distval))

                # Do not allow two votes to the same image per query descriptor.
                voted.append(imidx)

    try:

        # votescores is the number of votes per DB image.
        votescores = np.ma.array(np.array(votetable, dtype=np.float64), mask=False)

        # distscore is the average distance of matches between matched query and DB descriptors.
        distscores = np.ma.array(np.array(distsum, dtype=np.float64) / votescores, mask=False)

        # Excludes scores from images not voted. Uses masked arrays for that.
        if exclude_zero:
            votescores.mask[votescores==0] = True
            distscores.mask[votescores==0] = True

    except:
        sys.stderr.write("Problem creating score tables!\n")
        e_type, e_val, e_tb = sys.exc_info()
        traceback.print_exception(e_type, e_val, e_tb)
        return None, None, None

    # Here is a little trick: we esclude the most voted position because it is always the query image if it is present
    # in the DB. If the query image is NOT present in the DB, we remove the "exclude_first" flag.
    if exclude_first:
        mv = np.argmax(votescores)
        votescores.mask[mv] = True
        distscores.mask[mv] = True

    return votescores, distscores, dmatchlist


def normalize_scores(inarray, cvt_sim=False, min_val=-1):
    """ Perform min-max normalization, optionally converting similarity to dissimilarity, and vice versa.

    :param inarray: input array to be normalized;
    :param cvt_sim: If true, converts between similarity-dissimilarity. Conversion is done by subtracting the normalized
                    array from 1;
    :param min_val: (optional) minimum value to use in the min-max normalization. If less than 0, uses the minimum
                    computed from the array.
    :return: normalized array, indices of ordered normalized array.
    """

    if min_val < 0:
        mi = np.min(inarray)
    else:
        mi = min_val

    mx = np.max(inarray)

    norm_array = (inarray.astype(np.float64) - mi)/(mx - mi)
    score_order = norm_array.argsort()

    if cvt_sim:
        return 1 - norm_array, score_order
    else:
        return norm_array, score_order[::-1]
