# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 18:03:01 2014

@author: Federico Barabas
"""

import os
import numpy as np


def insertSuffix(filename, suffix, newExt=None):
    names = os.path.splitext(filename)
    if newExt is None:
        return names[0] + suffix + names[1]
    else:
        return names[0] + suffix + newExt


def mode(array):
    hist, bin_edges = np.histogram(array, bins=array.max() - array.min())
    hist_max = hist.argmax()
    return (bin_edges[hist_max + 1] + bin_edges[hist_max]) / 2


def overlaps(p1, p2, d):
    return max(abs(p1[1] - p2[1]), abs(p1[0] - p2[0])) <= d


def dropOverlapping(maxx, d):
    """We exclude from the analysis all the maxima in maxx that have their
    fitting windows overlapped, i.e., the distance between them is less than
    'd'."""

    noOverlaps = np.zeros(maxx.shape, dtype=int)  # Final array

    n = 0
    for i in np.arange(len(maxx)):
        def overlapFunction(x):
            return not(overlaps(maxx[i], x, d))
        overlapsList = map(overlapFunction, np.delete(maxx, i, 0))
        if all(overlapsList):
            noOverlaps[n] = maxx[i]
            n += 1

    return noOverlaps[:n]
