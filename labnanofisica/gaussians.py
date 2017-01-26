# -*- coding: utf-8 -*-
"""
Created on Sat May  7 21:41:17 2016

@author: Federico Barabas

Refs:
https://gist.github.com/andrewgiessel/6122739
"""

import numpy as np
from scipy import optimize
from scipy.special import jn
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def gauss(x, mu, sigma, A):
    return np.abs(A)*np.exp(-(x-mu)**2/2/sigma**2)


def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return gauss(x, mu1, sigma1, A1) + gauss(x, mu2, sigma2, A2)


def gauss_fwhm(x, fwhm):
    return np.exp(- 4 * np.log(2) * (x / fwhm)**2)


def best_gauss(x, x0, fwhm):
    """ Returns the closest gaussian function to an Airy disk centered in x0
    and a full width half maximum equal to fwhm."""
    return np.exp(- 4 * np.log(2) * (x - x0)**2 / fwhm**2)


def airy(x):
    return (2 * jn(1, 2 * np.pi * x) / (2 * np.pi * x))**2


def get_fwhm(wavelength, NA):
    ''' Gives the FWHM (in nm) for a PSF with wavelength in nm'''

    x = np.arange(-2, 2, 0.01)
    y = airy(x)

    # Fitting only inside first Airy's ring
    fit_int = np.where(abs(x) < 0.61)[0]

    fit_par, fit_var = curve_fit(gaussian, x[fit_int], y[fit_int], p0=0.5)

    return fit_par[0] * wavelength / NA


def airy_vs_gauss():

    wavelength = 670        # nm
    NA = 1.42

    x = np.arange(-2, 2, 0.01)
    y = airy(x)
    fw = get_fwhm(wavelength, NA)
    fit = best_gauss(x, 0, fw * NA / wavelength)

    print('FWHM is', np.round(fw))

    plt.plot(x, y, label='Airy disk')
    plt.plot(x, fit, label='Gaussian fit')
    plt.legend()
    plt.grid('on')
    plt.show()


def kernel(fwhm):
    """ Returns the kernel of a convolution used for finding objects of a
    full width half maximum fwhm (in pixels) in an image."""
    window = np.ceil(fwhm) + 3
#    window = int(np.ceil(fwhm)) + 2
    x = np.arange(0, window)
    y = x
    xx, yy = np.meshgrid(x, y, sparse=True)
    matrix = best_gauss(xx, x.mean(), fwhm) * best_gauss(yy, y.mean(), fwhm)
    matrix /= matrix.sum()
    return matrix


def xkernel(fwhm):
    window = np.ceil(fwhm) + 3
    x = np.arange(0, window)
    matrix = best_gauss(x, x.mean(), fwhm)
    matrix = matrix - matrix.sum() / matrix.size
    return matrix


class twoDSymmGaussian():

    def __init__(self, data):
        self.fit(data)

    def function(self, xdata, A, xo, yo, sigma, offset):
        (x, y) = xdata
        xo = float(xo)
        yo = float(yo)
        c = 2*sigma**2
        g = offset + A*np.exp(- ((x-xo)**2 + (y-yo)**2)/c)
        return g.ravel()

    def moments(self, data):
        """Returns (height, x, y, sigma, offset)
        the gaussian parameters of a 2D distribution by calculating its
        moments """
        offset = data.min()
        data -= offset
        total = data.sum()
        X, Y = np.indices(data.shape)
        x = (X*data).sum()/total
        y = (Y*data).sum()/total
        col = data[:, int(y)]
        width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
        row = data[int(x), :]
        width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
        height = data.max()
        return height, x, y, (width_x + width_y)/2, offset

    def fit(self, data):

        # Create x and y indices
        x = np.arange(0, data.shape[0], dtype=float)
        y = np.arange(0, data.shape[1], dtype=float)
        x, y = np.meshgrid(x, y)

        initial = self.moments(data)

        popt, pcov = optimize.curve_fit(self.function, (x, y), data.ravel(),
                                        p0=initial)
        self.popt = popt
        self.epopt = np.sqrt([pcov[i, i] for i in np.arange(pcov.shape[0])])


def twoDGaussian(xdata, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x, y) = xdata
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp(- (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) +
                                  c*((y-yo)**2)))
    return g.ravel()


def moments(data):
    """Returns (height, x, y, width_x, width_y, offset)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    offset = data.min()
    data -= offset
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y, offset
