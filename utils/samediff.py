#!/usr/bin/env python

"""
Functionality for performing same-different evaluation.

Details are given in:
- M. A. Carlin, S. Thomas, A. Jansen, and H. Hermansky, "Rapid evaluation of
  speech representations for spoken term discovery," in Proc. Interspeech,
  2011.

Author: Herman Kamper
Contact: h.kamper@sms.ed.ac.uk
Date: 2014
"""

import matplotlib
matplotlib.use('Agg')
from scipy.spatial.distance import pdist
import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import sys


#-----------------------------------------------------------------------------#
#                              SAMEDIFF FUNCTIONS                             #
#-----------------------------------------------------------------------------#

def average_precision(swsp_distances, swdp_distances, dw_distances, show_plot):
    """
    Calculate average precision and precision-recall breakeven.

    Return the average precision and precision-recall breakeven calculated
    using the same-word same-speaker distances `swsp_distances`, same-word
    different-speaker distances `swdp_distances` and true negative
    distances `dw_distances`.
    """
    distances = np.concatenate([swsp_distances, swdp_distances, dw_distances])
    sw_matches = np.concatenate([np.ones(len(swsp_distances)),
                                 np.ones(len(swdp_distances)),
                                 np.zeros(len(dw_distances))])
    swdp_matches = np.concatenate([np.zeros(len(swsp_distances)),
                                   np.ones(len(swdp_distances)),
                                   np.zeros(len(dw_distances))])

    # Sort from shortest to longest distance
    sorted_i = np.argsort(distances)
    distances = distances[sorted_i]
    sw_matches = sw_matches[sorted_i]
    swdp_matches = swdp_matches[sorted_i]

    # Calculate precision
    sw_precision = np.cumsum(sw_matches)/np.arange(1, len(sw_matches) + 1)
    sw_precision2 = np.cumsum(sw_matches)/np.arange(1, len(sw_matches) + 1)

    # Calculate average precision: the multiplication with matches and division
    # by the number of positive examples is to not count precisions at the same
    # recall point multiple times.
    sw_average_precision = np.sum(sw_precision * sw_matches) / (
        len(swsp_distances) + len(swdp_distances))
    average_precision = np.sum(sw_precision * swdp_matches) / len(swdp_distances)

    # Calculate recall
    sw_recall = np.cumsum(sw_matches)/(len(swsp_distances) + len(swdp_distances))
    swdp_recall = np.cumsum(swdp_matches)/len(swdp_distances)

    # More than one precision can be at a single recall point, take the max one
    for n in range(len(sw_recall) - 2, -1, -1):
        sw_precision[n] = max(sw_precision[n], sw_precision[n + 1])
    for n in range(len(swdp_recall) - 2, -1, -1):
        sw_precision2[n] = max(sw_precision2[n], sw_precision2[n + 1])

    # Calculate precision-recall breakeven
    sw_prb_i = np.argmin(np.abs(sw_recall - sw_precision))
    sw_prb = (sw_recall[sw_prb_i] + sw_precision[sw_prb_i])/2.
    swdp_prb_i = np.argmin(np.abs(swdp_recall - sw_precision2))
    swdp_prb = (swdp_recall[swdp_prb_i] + sw_precision2[swdp_prb_i])/2.

    show_plot = True
    if show_plot:
        plt.plot(swdp_recall, sw_precision2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        #plt.savefig("/afs/inf.ed.ac.uk/user/s16/s1680167/pr_utd3.png", format='png')

    return average_precision, sw_average_precision, swdp_prb, sw_prb


def generate_matches_array(labels, speakers=[]):
    """
    Return an array of bool in the same order as the distances from
    `scipy.spatial.distance.pdist` indicating whether a distance is for
    matching or non-matching labels.
    """
    N = len(labels)
    word_matches = np.zeros(N*(N - 1)/2, dtype=np.bool)
    speaker_matches = None

    # For every distance, mark whether it is a true match or not
    cur_matches_i = 0
    for n in range(N):
        cur_label = labels[n]
        word_matches[cur_matches_i:cur_matches_i + (N - n) - 1] = np.asarray(labels[n + 1:]) == cur_label
        cur_matches_i += N - n - 1

    if speakers != [] and len(speakers) == len(labels):
        speaker_matches = np.zeros(N*(N - 1)/2, dtype=np.bool)
        cur_matches_i = 0
        for n in range(N):
            cur_speaker = speakers[n]
            speaker_matches[cur_matches_i:cur_matches_i + (N - n) - 1] = np.asarray(speakers[n + 1:]) == cur_speaker
            cur_matches_i += N - n - 1

    return word_matches, speaker_matches


def fixed_dim(X, labels, metric="cosine", show_plot=False):
    """
    Return average precision and precision-recall breakeven calculated on
    fixed-dimensional set `X`.

    `X` contains the fixed-dimensional data items as row vectors.
    """
    N, D = X.shape
    matches = generate_matches_array(labels)
    distances = pdist(X, metric)
    return average_precision(distances[matches == True], distances[matches == False], show_plot)


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("labels_fn", help="file of labels")
    parser.add_argument(
        "distances_fn",
        help="file providing the distances between each pair of labels in the same order as "
        "`scipy.spatial.distance.pdist`"
        )
    parser.add_argument(
        "--binary_dists", dest="binary_dists", action="store_true",
        help="distances are given in float32 binary format "
        "(default is to assume distances are given in text format)"
        )
    parser.set_defaults(binary_dists=False)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    
    args = check_argv()

    # Read labels
    labels = [i.strip().split()[0] for i in open(args.labels_fn)]
    try:
        speakers = [i.strip().split()[1] for i in open(args.labels_fn)]
    except IndexError:
        speakers = []
    N = len(labels)

    # Read distances
    print "Start time: " + str(datetime.datetime.now())
    if args.binary_dists:
        print "Reading distances from binary file:", args.distances_fn
        distances_vec = np.fromfile(args.distances_fn, dtype=np.float32)
    else:
        print "Reading distances from text file:", args.distances_fn
        distances_vec = np.fromfile(args.distances_fn, dtype=np.float32, sep="\n")
    if np.isnan(np.sum(distances_vec)):
        print "Warning: Distances contain nan"
        # distances_vec = np.nan_to_num(distances_vec)
        # distances_vec[np.where(np.isnan(distances_vec))] = np.inf

    # print "Reading distances from:", args.distances_fn
    # distances_vec = np.zeros(N*(N - 1)/2)
    # i_dist_vec = 0
    # for line in open(args.distances_fn):
    #     distances_vec[i_dist_vec] = float(line.strip())
    #     i_dist_vec += 1

    # Calculate average precision
    print "Calculating statistics."
    word_matches, speaker_matches = generate_matches_array(labels, speakers)
    ap, sw_ap, prb, sw_prb = average_precision(
        distances_vec[np.logical_and(word_matches, speaker_matches)],
        distances_vec[np.logical_and(word_matches, speaker_matches == False)],
        distances_vec[word_matches == False],
        False)
    print "SWDP Average precision:", ap
    print "SWDP Precision-recall breakeven:", prb
    print "SW Average precision:", sw_ap
    print "SW Precision-recall breakeven:", sw_prb
    print "End time: " + str(datetime.datetime.now())


if __name__ == "__main__":
    main()
