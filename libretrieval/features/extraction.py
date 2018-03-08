#/usr/bin/env python
# -*- coding: utf-8 -*-

"""
objret.keypoints.extraction.py

Extraction and Desacription of local features on image files. Use OpenCV's implementation of local feature extractors
and descriptors.

"""

import sys
import traceback
import cv2
import numpy as np
import time


def local_feature_detection(img, detetype, kmax=500):
    """ Sparsely detects local features in an image.

    OpenCV implementation of various detectors.

    :param img: input image;
    :param detetype: type of detector {SURF, SIFT, ORB, BRISK}.
    :param kmax: maximum number of keypoints to return. The kmax keypoints with largest response are returned;

    :return: detected keypoins; detection time;
    """

    try:
        if detetype == "SURF":
            surf = cv2.SURF()
            st_t = time.time()
            keypoints = surf.detect(img)
            ed_t = time.time()

            if kmax != -1:
                keypoints = keypoints[0:kmax]

        elif detetype == "SIFT":
            sift = cv2.SIFT(nfeatures=kmax)
            st_t = time.time()
            keypoints = sift.detect(img)
            ed_t = time.time()

        elif detetype == "ORB":
            orb = cv2.ORB(nfeatures=kmax)
            st_t = time.time()
            keypoints = orb.detect(img)
            ed_t = time.time()

        elif detetype == "BRISK":
            brisk = cv2.BRISK()
            st_t = time.time()
            keypoints = brisk.detect(img)
            ed_t = time.time()

            keypoints = keypoints[0:kmax]

        elif detetype == "Dense":
            keypoints = detect_dense_keypoints(img)

        else:
            surf = cv2.SURF()
            st_t = time.time()
            keypoints = surf.detect(img)
            ed_t = time.time()

        det_t = ed_t - st_t
        return keypoints, det_t

    except Exception as e:
        sys.stderr.write("Failure in detecting features\n")
        e_type, e_val, e_tb = sys.exc_info()
        traceback.print_exception(e_type, e_val, e_tb)
        return [], -1


def local_feature_description(img, keypoints, desctype):
    """ Describes the given keypoints of an image.

    OpenCV implementation of various descriptors.

    :param img: input image;
    :param keypoints: computed keypoints;
    :param desctype: type of descriptor {SURF, SIFT, ORB, BRISK, RootSIFT}.

    :return: computed features, description time.
    """

    try:
        if desctype == "SURF":
            surf = cv2.SURF()
            st_t = time.time()
            __, features = surf.compute(img, keypoints)
            ed_t = time.time()

        elif desctype == "SIFT":
            sift = cv2.SIFT()
            st_t = time.time()
            __, features = sift.compute(img, keypoints)
            ed_t = time.time()

        elif desctype == "ORB":
            orb = cv2.ORB()
            st_t = time.time()
            __, features = orb.compute(img, keypoints)
            ed_t = time.time()

        elif desctype == "BRISK":
            brisk = cv2.BRISK()
            st_t = time.time()
            __, features = brisk.compute(img, keypoints)
            ed_t = time.time()

        elif desctype == "RootSIFT":
            eps = 0.00000001
            sift = cv2.SIFT()
            st_t = time.time()
            __, features = sift.compute(img, keypoints)

            features /= (np.sum(features, axis=1, keepdims=True) + eps)
            features = np.sqrt(features)

            ed_t = time.time()

        else:
            surf = cv2.SURF()
            st_t = time.time()
            __, features = surf.compute(img, keypoints, descriptors=features)
            ed_t = time.time()

        dsc_t = ed_t - st_t
        return features, dsc_t

    except:
        sys.stderr.write("Failure in detecting features\n")
        e_type, e_val, e_tb = sys.exc_info()
        traceback.print_exception(e_type, e_val, e_tb)
        return [], -1

def local_feature_detection_and_description(imgpath, detetype, desctype, kmax=500, img=[]):
    """ Given a path or an image, detects and describes local features.

    :param imgpath: path to the image
    :param detetype: type of detector {SURF, SIFT, ORB, BRISK}.
    :param desctype: type of descriptor {SURF, SIFT, ORB, BRISK, RootSIFT}.
    :param kmax: maximum number of keypoints to return. The kmax keypoints with largest response are returned;
    :param img: (optional) input image. If not present, loads the image from imgpath.

    :return: detected keypoints, described features, detection time, description time.
    """

    if img == []:
        img = cv2.imread(imgpath)

    #print imgpath
    try:
        keyps, det_t = local_feature_detection(img, detetype, kmax)
        if not keyps:
            raise ValueError

        feat, dsc_t = local_feature_description(img, keyps, desctype)
        if feat is []:
            raise ValueError

        return keyps, feat, det_t, dsc_t

    except ValueError:
        return [], [], -1, -1

