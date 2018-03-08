#/usr/bin/env python
# -*- coding: utf-8 -*-

"""
objret.features.keypoints.py

Auxiliary functions to manage image keypoints. Allows conversion between OpenCV's KeyPoint class and simpler, array
based structures, as well as other utilitary functions.

"""

import sys
import os
import traceback

import numpy
import cv2


def keypoints_from_array(keyparray):
    """ Creates a list of OpenCV's KeyPoint structures, given an array.

    Each row of the input array is organized as:
    <px> <py> <size> <angle> <response> <octave> <id>,

    were px and py are the xy coord. of the KeyPoint, size, response and octave the corresponding attributes and
    id and optional image id.

    :param keyparray: array of keypoints;

    :return: list of cv2.KeyPoint
    """

    keypoints = []

    for row in keyparray:

        px = row[0]
        py = row[1]
        sz = row[2]
        ang = row[3]
        resp = row[4]
        oct = row[5]
        cid = row[6]

        keyp = cv2.KeyPoint(x=px, y=py, _size=sz, _angle=ang, _response=resp, _octave=oct, _class_id=cid)
        keypoints.append(keyp)

    return keypoints


def keypoints_to_array(keypoints):
    """ Converts a list of cv2.KeyPoint to an array of their attributes.

    Each row of the output array is organized as:
    <px> <py> <size> <angle> <response> <octave> <id>,

    were px and py are the xy coord. of the KeyPoint, size, response and octave the corresponding attributes and
    id and optional image id.

    :param keypoints: list of cv2.KeyPoint.
    :return: array of keypoint attributes.
    """

    aux = []
    for keyp in keypoints:

        px, py = keyp.pt
        sz = keyp.size
        ang = keyp.angle
        resp = keyp.response
        oct = keyp.octave
        cid = keyp.class_id

        aux.append([px, py, sz, ang, resp, oct, cid])

    keyparray = numpy.array(aux, dtype=numpy.float32)

    return keyparray

def get_matched_keypoints(query_keyps, train_keyps, dmatchlist):
    """ Given query and train keypoins, and a list of DMatch, get lists of the matched query and train keypoints.

    Assumes each query keypoint has AT MOST one matched train keypoint (KNN = 1).

    :param query_keyps: input unorganzied query keypoints
    :param train_keyps: input unorganized train keypoints
    :param dmatchlist: list of lists containing cv2.DMatch.

    :return: query matching keypoints, train matching keypoints.
    """

    matched_query_keyps = []
    matched_train_keyps = []

    for dmatch in dmatchlist:

        try:
            qidx = dmatch.queryIdx
            tidx = dmatch.trainIdx
        except AttributeError:
            qidx = dmatch[0].queryIdx
            tidx = dmatch[0].trainIdx

        matched_query_keyps.append(query_keyps[qidx])
        matched_train_keyps.append(train_keyps[tidx])

    #print "    > Got matched keypoints."
    return matched_query_keyps, matched_train_keyps
