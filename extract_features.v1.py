#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import sys
import argparse
import glob
import errno

import cv2

from objret.features.extraction import *
from objret.features.keypoints import *

from objret.io import read_array_file
from objret.io import write_array_file
from objret.io import write_array_bin_file

verbose = False

def safe_create_dir(dir):
    try:
        os.makedirs(dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
        else:
            if verbose: print "Directory already exists!"
            return

def write_feat_indexing(filepath, feat_indexing):

    f = open(filepath, 'w')

    imname, featn, feati = feat_indexing[0]
    f.write("{0:<100s} {1:d} {2:d}".format(imname, featn, feati))

    for row in feat_indexing[1:]:

        imname, featn, feati = row
        f.write("\n{0:<100s} {1:d} {2:d}".format(imname, featn, feati))

    f.close()

    return

def extract_features_on_dir(imgdir, outdir, detector, descriptor, limit, interval, save, outtype, prefix):

    safe_create_dir(outdir)

    if verbose: print "Extracting features on dir: ", imgdir
    if verbose: print "Output directory: ", outdir
    if verbose: print "Detector: ", detector
    if verbose: print "Descriptor: ", descriptor
    if verbose: print "Maximum number of features: ", limit

    imgplist = glob.glob(imgdir + "*.jpg")
    imgplist.sort()

    nimg = len(imgplist)

    if interval[0] < 0:
        start_img = 0
    else:
        start_img = interval[0]

    if interval[1] > nimg or interval[1] < 0:
        stop_img = nimg
    else:
        stop_img = interval[1]

    allfeat = []
    allkeyp = []
    feat_indexing = []
    count = 0
    if verbose: print "\n--- Starting extraction: from image {0:d} to {1:d} ---\n".format(start_img, stop_img)
    for i in xrange(start_img, stop_img):

        imgpath = imgplist[i]
        imgname = os.path.basename(imgpath)
        basename = os.path.splitext(imgname)[0]

        # 0 is for binary output file
        if outtype == 0:
            featoutfpath = "{0:s}{1:s}.bfv".format(outdir, basename)
            kpoutfpath = "{0:s}{1:s}.bkp".format(outdir, basename)
        else:
            featoutfpath = "{0:s}{1:s}.fv".format(outdir, basename)
            kpoutfpath = "{0:s}{1:s}.kp".format(outdir, basename)

        if verbose: print "Extracting #{0:d}: {1:s}".format(i, imgname)

        if not os.path.isfile(featoutfpath) and not os.path.isfile(kpoutfpath):

            keypoints, features, det_t, dsc_t = local_feature_detection_and_description(imgpath,
                                                                                        detector,
                                                                                        descriptor,
                                                                                        kmax=limit,
                                                                                        img=[])

            # Converts OpenCV's KeyPoint structure to numpy array for storage purposes
            keyparray = keypoints_to_array(keypoints)

            if features is not None and len(features) > 0:

                count += features.shape[0]
                feat_indexing.append([imgname, features.shape[0], count])

                if save == 'individual' or save == 'both':

                    if outtype == 0:
                        write_array_bin_file(featoutfpath, features)
                        write_array_bin_file(kpoutfpath, keyparray)
                    else:
                        write_array_file(featoutfpath, features)
                        write_array_file(kpoutfpath, keyparray, outfmt='%.4f')

                allfeat.append(features)
                allkeyp.append(keyparray)
                if verbose: print "    Done! (Detection: {0:0.4f}s / Description: {1:0.4f}s)".format(det_t, dsc_t)
                if verbose: print "    Outfile is: {0:s}\n".format(featoutfpath)

            else:

                if verbose: print "No features found!"

        else:
            if verbose: print "    Output file {0:s} already exists!\n".format(featoutfpath)
            if save == 'all' or save == 'both':
                allfeat.append(read_array_file(featoutfpath))
                allkeyp.append(read_array_file(kpoutfpath))

    if (save == 'all' or save == 'both') and len(allfeat) > 0:

        if outtype == 0:
            allfoutpath = "{0:s}{1:s}.bfv".format(outdir, prefix)
            allkoutpath = "{0:s}{1:s}.bkp".format(outdir, prefix)
            if verbose: print "Saving all features: \n   >", allfoutpath, "\n    >", allkoutpath
            write_array_bin_file(allfoutpath, allfeat)
            write_array_bin_file(allkoutpath, allkeyp)
        else:
            allfoutpath = "{0:s}{1:s}.fv".format(outdir, prefix)
            allkoutpath = "{0:s}{1:s}.kp".format(outdir, prefix)
            if verbose: print "Saving all features and keypoints: \n   >", allfoutpath, "\n   >", allkoutpath
            write_array_file(allfoutpath, allfeat)
            write_array_file(allkoutpath, allkeyp, outfmt='%.5f')

    write_feat_indexing("{0:s}{1:s}_indexing.txt".format(outdir, prefix), feat_indexing)



    if verbose: print "--- Done ---"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("imgdir",
                        help="Directory containing the images to extract features from.",
                        type=str)

    parser.add_argument("outdir",
                        help="Output directory for feature vector (.fv) files.",
                        type=str)

    parser.add_argument("--detector", "-d",
                        help="Local Feature detector used.",
                        type=str,
                        choices=["SURF", "SIFT", "ORB", "DENSE", "BRISK"],
                        default="SURF")

    parser.add_argument("--descriptor", "-f",
                        help="Local descriptor used.",
                        type=str,
                        choices=["SURF", "SIFT", "ORB", "RootSIFT", "RootSURF", "BRISK"],
                        default="SURF")

    parser.add_argument("--limit", "-l",
                        help="Maximum number of keypoints to be detected by sparse detectors.",
                        type=int,
                        default=500)

    parser.add_argument("--interval", "-i",
                        help="Pair of number enumerating the images to start and end extraction. "
                             "Expects a start value and an end value. For example, -i 10 20"
                             "extracts features from image 10 to image 19.",
                        type=int,
                        nargs=2,
                        default=[-1, -1])

    parser.add_argument("--save", "-s",
                        help="Indicates the files to save. 'individual' saves one feature vector "
                             "file for each image. 'all' saves one file for all the features "
                             "extracted from multiple images. 'both' saves both cases. Default "
                             "is 'individual'.",
                        type=str,
                        choices=["individual", "all", "both"],
                        default="individual")

    parser.add_argument("--outtype", "-t",
                        help="Type of the output file. 0 for binary files, 1 for txt files."
                             "Default is binary",
                        type=int,
                        choices=[0, 1],
                        default=0)

    parser.add_argument("--prefix", "-p",
                        help="Optional prefix for output files. Default is empty ",
                        type=str,
                        default="")

    parser.add_argument("--verbose", "-v",
                        help="Printing of additional info",
                        action="store_true")


    args = parser.parse_args()

    imgdir = args.imgdir
    outdir = args.outdir
    detector = args.detector
    descriptor = args.descriptor
    limit = args.limit
    interval = args.interval
    save = args.save
    outtype = args.outtype
    prefix = args.prefix
    verbose = args.verbose

    if imgdir[-1] != "/":
        imgdir += "/"

    if outdir[-1] != "/":
        outdir += "/"

    extract_features_on_dir(imgdir, outdir, detector, descriptor, limit, interval, save, outtype, prefix)