# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 14:53:28 2016

@author: Luciano Masullo, Federico Barabas
"""

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import skimage.filters as filters
from skimage.transform import probabilistic_hough_line

from pyqtgraph.Qt import QtCore, QtGui

from labnanofisica.ringfinder.neurosimulations import simAxon


def pearson(a, b):
    """2D pearson coefficient of two matrixes a and b"""

    # Subtracting mean values
    an = a - np.mean(a)
    bn = b - np.mean(b)

    # Vectorized versions of c, d, e
    c_vect = an*bn
    d_vect = an*an
    e_vect = bn*bn

    # Finally get r using those vectorized versions
    r_out = np.sum(c_vect)/np.sqrt(np.sum(d_vect)*np.sum(e_vect))

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


def getDirection(data, sigma, pxSize, minLen):
    """Returns the direction (angle) of the neurite in the image data.

    sigma: gaussian filter sigma to blur the image, in nm
    pxSize: size of px in nm
    minLen: minimum  line length in nm."""

    # gaussian filter to get low resolution image
    sigma_px = sigma/pxSize
    img = ndi.gaussian_filter(data, sigma_px)

    # TO DO: good cond on intensity
    if np.sum(img) < 10:
        return None

    else:
        # binarization of image
        thresh = filters.threshold_otsu(img)
        binary = img > thresh

        minLen /= pxSize    # minLen in pxs
        meanAngle, stdAngle = linesFromBinary(binary, minLen)

        if meanAngle is not None:

            # if the std is too high it's probably the case of flat angles,
            # i.e., 181, -2, 0.3, -1, 179
            # TO DO: find optimal threshold, 40 is arbitrary
            if stdAngle > 40:
                print('std too big, will rotate data and try again')
                binary = np.rot90(binary)
                meanAngle, stdAngle = linesFromBinary(binary, minLen)
                return meanAngle - 90
            else:
                return meanAngle

        else:
            return None


def linesFromBinary(binaryData, minLen):

    # find edges
    edges = filters.sobel(binaryData)

    # get directions
    lines = probabilistic_hough_line(edges, threshold=10, line_length=minLen,
                                     line_gap=3)

    # allocate angleArr which will have the angles of the lines
    angleArr = []

    if lines == []:
        return None, None

    else:
        for line in lines:
            p0, p1 = line

            # get the m coefficient of the lines and the angle
            if p1[1] == p0[1]:
                angle = 90
            else:
                m = (p1[0] - p0[0])/(p1[1] - p0[1])
                angle = (180/np.pi)*np.arctan(m)

            angleArr.append(angle)

        # calculate mean angle and its standard deviation
        print('angleArr is {}'.format(np.around(angleArr, 1)))

        return np.mean(angleArr), np.std(angleArr)


def corrMethod(data, thres, sigma, pxSize, minLen, thStep, deltaTh, wvlen,
               sinPow, developer=False):
    """Searches for rings by correlating the image data with a given
    sinusoidal pattern

    data: 2D image data
    thres: discrimination threshold for the correlated data.
    sigma: gaussian filter sigma to blur the image, in nm
    pxSize: size of px in nm
    minLen: minimum line length in nm.
    thStep: angular step size
    deltaTh: maximum pattern rotation angle for correlation matching
    wvlen: wavelength of the ring pattern, in nm
    sinPow: power of the pattern function

    returns:

    corrMax: the maximum (in function of the rotated angle) correlation value
    at the image data
    thetaMax: simulated axon's rotation angle with maximum correlation value
    phaseMax: simulated axon's phase with maximum correlation value at thetaMax
    rings (bool): ring presence"""

    # line angle calculated
    th0 = getDirection(data, sigma, pxSize, minLen)

    if th0 is None:
        return th0, 0, 0, 0, 0, False
    else:
        print('line angle is {}'.format(np.around(th0, 1)))
        subImgSize = np.shape(data)[0]

        # set the angle range to look for a correlation, 179 is added
        # because of later corrAngle's expansion
        if developer:
            theta = np.arange(0, 180, thStep)
        else:
            theta = np.arange(th0 - deltaTh, th0 + deltaTh, thStep)

        # phase steps are set to 20, TO DO: explore this parameter
        phase = np.arange(0, 21, 1)

        corrPhase = np.zeros(np.size(phase))
        corrPhaseArg = np.zeros(np.size(theta))
        corrTheta = np.zeros(np.size(theta))

        wvlen = wvlen/pxSize  # wvlen in px

        # for now we correlate with the full sin2D pattern
        for t in np.arange(len(theta)):
            for p in phase:
                # creates simulated axon
                axonTheta = simAxon(subImgSize, wvlen, theta[t], p*.025, a=0,
                                    b=sinPow).data

                # saves correlation for the given phase p
                corrPhase[p] = pearson(data, axonTheta)

            # saves the correlation for the best p, and given angle i
            corrTheta[t - 1] = np.max(corrPhase)
            corrPhaseArg[t - 1] = .025*np.argmax(corrPhase)

        # get theta, phase and correlation with greatest correlation value
        i = np.argmax(corrTheta)
        thetaMax = theta[i]
        phaseMax = corrPhaseArg[i]
        corrMax = np.max(corrTheta)

        rings = corrMax > thres

        return th0, corrTheta, corrMax, thetaMax, phaseMax, rings


def FFTMethod(data, thres=0.4):
    """A method for actin/spectrin ring finding. It performs FFT 2D analysis
    and looks for maxima at 180 nm in the frequency spectrum."""

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


def pointsMethod(self, data, thres=.3):
    """A method for actin/spectrin ring finding. It finds local maxima in the
    image (points) and then if there are three or more in a row considers that
    to be rings."""

    points = peak_local_max(data, min_distance=6, threshold_rel=thres)
    points = firstNmax(points, data, N=7)

    D = []

    if points == []:
        rings = False

    else:
        dmin = 8
        dmax = 11

        # look up every point
        for i in np.arange(0, np.shape(points)[0]-1):
            # calculate the distance of every point to the others
            for j in np.arange(i + 1, np.shape(points)[0]):
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


def setGrid(viewbox, image, n=10):

    shape = image.shape

    pen = QtGui.QPen(QtCore.Qt.yellow, 1, QtCore.Qt.SolidLine)
    rect = QtGui.QGraphicsRectItem(0, 0, shape[0], shape[1])
    rect.setPen(pen)
    viewbox.addItem(rect)

    for i in np.arange(0, n - 1):
        cx = (shape[0]/n)*(i + 1)
        cy = (shape[1]/n)*(i + 1)
        linex = QtGui.QGraphicsLineItem(0, cx, shape[0], cx)
        liney = QtGui.QGraphicsLineItem(cy, 0, cy, shape[1])
        linex.setPen(pen)
        liney.setPen(pen)
        viewbox.addItem(linex)
        viewbox.addItem(liney)


class Grid:

    def __init__(self, viewbox, image, n=10):
        self.vb = viewbox
        self.n = n
        self.lines = []
        self.draw(image)

    def draw(self, image):
        shape = image.shape
        pen = QtGui.QPen(QtCore.Qt.yellow, shape[0]//250, QtCore.Qt.SolidLine)
        self.rect = QtGui.QGraphicsRectItem(0, 0, shape[0], shape[1])
        self.rect.setPen(pen)
        self.vb.addItem(self.rect)
        self.lines.append(self.rect)

        for i in np.arange(0, self.n - 1):
            cx = (shape[0]/self.n)*(i + 1)
            cy = (shape[1]/self.n)*(i + 1)
            linex = QtGui.QGraphicsLineItem(0, cx, shape[0], cx)
            liney = QtGui.QGraphicsLineItem(cy, 0, cy, shape[1])
            linex.setPen(pen)
            liney.setPen(pen)
            self.vb.addItem(linex)
            self.vb.addItem(liney)
            self.lines.append(linex)
            self.lines.append(liney)

#    def clear(self):
#
#        for line in self.lines:
#            self.vb.removeItem(line)
