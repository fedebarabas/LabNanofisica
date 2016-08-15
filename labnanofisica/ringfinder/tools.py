# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 14:53:28 2016

@author: Luciano Masullo, Federico Barabas
"""

import numpy as np
from skimage.feature import peak_local_max


def corr2(a, b):
    """2D pearson coefficient of two matrixes a and b"""

    # Calculating mean values
    AM = np.mean(a)
    BM = np.mean(b)

    # Vectorized versions of c,d,e

    c_vect = (a-AM)*(b-BM)
    d_vect = (a-AM)**2
    e_vect = (b-BM)**2

    # Finally get r using those vectorized versions
    r_out = np.sum(c_vect)/float(np.sqrt(np.sum(d_vect)*np.sum(e_vect)))

    return r_out


def cosTheta(a, b):
    """Angle between two vectors a and b"""

    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0

    cosTheta = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

    return cosTheta


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1, 2)
               .reshape(-1, nrows, ncols))


def firstNmax(coord, image, N):
    """Returns the first N max in an image from an array of coord of the max
       in the image"""

    if np.shape(coord)[0] < N:
        return []
    else:
        aux = np.zeros(np.shape(coord)[0])
        for i in np.arange(np.shape(coord)[0]):
            aux[i] = image[coord[i, 0], coord[i, 1]]

        auxmax = aux.argsort()[-N:][::-1]

        coordinates3 = []
        for i in np.arange(0, N):
            coordinates3.append(coord[auxmax[i]])

        coord3 = np.asarray(coordinates3)

        return coord3


def arrayExt(array):
    """Extends an array in a specific way"""

    y = array[::-1]
    z = []
    z.append(y)
    z.append(array)
    z.append(y)
    z = np.array(z)
    z = np.reshape(z, 3*np.size(array))

    return z


def FFT2(self, data, thres=0.4):
    """FFT 2D analysis of actin rings. Looks for maxima at 180 nm in the
    frequency spectrum"""

    # calculate new fft2
    fft2output = np.real(np.fft.fftshift(np.fft.fft2(data)))

    # take abs value and log10 for better visualization
    fft2output = np.abs(np.log10(fft2output))

    # calculate local intensity maxima
    coord = peak_local_max(fft2output, min_distance=2, threshold_rel=thres)

    # take first 3 max
    coord = firstNmax(coord, fft2output, N=3)

    # size of the subimqge of interest
    A = np.shape(data)[0]

    # max and min radius in pixels, 9 -> 220 nm, 12 -> 167 nm
    rmin, rmax = (9, 12)

    # auxarrays: ringBool, D

    # ringBool is checked to define wether there are rings or not
    ringBool = []

    # D saves the distances of local maxima from the centre of the fft2
    D = []

    # loop for calculating all the distances d, elements of array D
    for i in np.arange(0, np.shape(coord)[0]):
        d = np.linalg.norm([A/2, A/2], coord[i])
        D.append(d)
        if A*(rmin/100) < d < A*(rmax/100):
            ringBool.append(1)

    # condition for ringBool: all elements d must correspond to
    # periods between 170 and 220 nm
    rings = np.sum(ringBool) == np.shape(coord)[0]-1 and np.sum(ringBool) > 0

    return fft2output, coord, (rmin, rmax), rings


def points(self, data, thres=.3):
    """Finds local maxima in the image (points) and then if there are
    three or more in a row considers that to be actin rings"""

    points = peak_local_max(data, min_distance=6, threshold_rel=thres)
    points = firstNmax(points, data, N=7)

    if points == []:
        return 0

    dmin = 8
    dmax = 11

    D = []

    # look up every point
    for i in np.arange(0, np.shape(points)[0]-1):
        # calculate the distance of every point to the others
        for j in np.arange(i+1, np.shape(points)[0]):
            d1 = np.linalg.norm(points[i], points[j])
            # if there are two points at the right distance then
            if dmin < d1 < dmax:
                for k in np.arange(0, np.shape(points)[0]-1):
                    # check the distance between the last point
                    # and the other points in the list
                    if k != i & k != j:
                        d2 = np.linalg.norm(points[j], points[k])

                    else:
                        d2 = 0

                    # calculate the angle between vector i-j
                    # and j-k with i, j, k points
                    v1 = points[i]-points[j]
                    v2 = points[j]-points[k]
                    t = cosTheta(v1, v2)

                    # if point k is at right distance from point j and
                    # the angle is flat enough
                    if dmin < d2 < dmax and np.abs(t) > 0.8:
                        # save the three points and plot the connections
                        D.append([points[i], points[j], points[k]])

                    else:
                        pass

    rings = len(D) > 0

    return points, D, rings
