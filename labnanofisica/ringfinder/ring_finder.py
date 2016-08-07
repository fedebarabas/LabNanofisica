# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 13:41:48 2014

@authors: Luciano Masullo
"""

from scipy import ndimage as ndi
#import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
# from skimage.feature import canny
from skimage import filters
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
# from skimage import data, img_as_float

# import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from labnanofisica.ringfinder.neurosimulations import simAxon

# import labnanofisica.sm.maxima as maxima

# from scipy import ndimage as ndi
# import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, sobel

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
# import pyqtgraph.console
# from pyqtgraph.dockarea import Dock, DockArea
# import pyqtgraph.ptime as ptime
from tkinter import Tk, filedialog, simpledialog

# from scipy.special import erf
# from scipy.optimize import minimize
# from scipy.ndimage import label
# from scipy.ndimage.filters import convolve, maximum_filter
# from scipy.ndimage.measurements import maximum_position, center_of_mass

# import labnanofisica.sm.tools as tools
# import labnanofisica.gaussians as gaussians

# import warnings
# warnings.filterwarnings("error")


def getFilename(title, types, initialdir=None):
    try:
        root = Tk()
        root.withdraw()
        filename = filedialog.askopenfilename(title=title, filetypes=types,
                                              initialdir=initialdir)
        root.destroy()
        return filename
    except OSError:
        print("No file selected!")


def corr2(a, b):

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


def dist(a, b):

    out = np.linalg.norm(a-b)

    return out


def cosTheta(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0

    cosTheta = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

    return cosTheta


def firstNmax(coord, image, N):
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

    y = array[::-1]
    z = []
    z.append(y)
    z.append(array)
    z.append(y)
    z = np.array(z)
    z = np.reshape(z, 3*np.size(array))

    return z


class RingAnalizer(QtGui.QMainWindow):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.setWindowTitle('RingAnalizer')

        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)

        self.setFixedSize(1275, 800)

        self.FFT2Button = QtGui.QPushButton('FFT 2D')
        self.corrButton = QtGui.QPushButton('Correlation')
        self.corr2thrEdit = QtGui.QLineEdit('0.1')
        self.thetaStepEdit = QtGui.QLineEdit('3')
        self.deltaAngleEdit = QtGui.QLineEdit('30')
        self.sinPowerEdit = QtGui.QLineEdit('2')
        self.pointsButton = QtGui.QPushButton('Points')
        self.loadimageButton = QtGui.QPushButton('Load Image')
        self.pxSizeEdit = QtGui.QLineEdit('20')
        self.stormCropButton = QtGui.QPushButton('STORM crop')
        self.filterImageButton = QtGui.QPushButton('Filter Image')
        self.fftThrEdit = QtGui.QLineEdit('0.6')
        self.pointsThrEdit = QtGui.QLineEdit('0.6')
        self.roiSizeEdit = QtGui.QLineEdit('50')
        self.dirButton = QtGui.QPushButton('Get direction')
        self.sigmaEdit = QtGui.QLineEdit('3')
        self.intThrButton = QtGui.QPushButton('Intensity threshold')
        self.intThrEdit = QtGui.QLineEdit('3')
        self.lineLengthEdit = QtGui.QLineEdit('.2')
        self.wvlenEdit = QtGui.QLineEdit('180')

        self.roiLabel = QtGui.QLabel('ROI size (px)')
        self.corr2thrLabel = QtGui.QLabel('Correlation threshold')
        self.thetaStepLabel = QtGui.QLabel('Angular step (°)')
        self.deltaAngleLabel = QtGui.QLabel('Delta Angle (°)')
        self.sigmaLabel = QtGui.QLabel('Sigma of gaussian filter')
        self.sinPowerLabel = QtGui.QLabel('Sinusoidal pattern power')
        self.pxSizeLabel = QtGui.QLabel('Pixel size (nm)')
        self.intThrLabel = QtGui.QLabel('# of times from mean intensity')
        self.lineLengthLabel = QtGui.QLabel('Direction lines length')
        # Direction lines lengths are expressed in fraction of subimg size
        self.wvlenLabel = QtGui.QLabel('wvlen of corr pattern (nm)')

        # Widgets' layout
        self.layout = QtGui.QGridLayout()
        self.cwidget.setLayout(self.layout)
        self.layout.setColumnStretch(1, 1)

        self.imagegui = ImageGUI(self)
        self.layout.addWidget(self.imagegui, 0, 0, 18, 10)
        self.layout.addWidget(self.loadimageButton, 0, 11, 1, 2)
        self.layout.addWidget(self.pxSizeLabel, 1, 11, 1, 1)
        self.layout.addWidget(self.pxSizeEdit, 1, 12, 1, 1)
        self.layout.addWidget(self.roiLabel, 2, 11, 1, 1)
        self.layout.addWidget(self.roiSizeEdit, 2, 12, 1, 1)

        self.layout.addWidget(self.corrButton, 3, 11, 1, 2)
        self.layout.addWidget(self.corr2thrLabel, 4, 11, 1, 1)
        self.layout.addWidget(self.corr2thrEdit, 4, 12, 1, 1)
        self.layout.addWidget(self.wvlenLabel, 5, 11, 1, 1)
        self.layout.addWidget(self.wvlenEdit, 5, 12, 1, 1)
        self.layout.addWidget(self.sinPowerLabel, 6, 11, 1, 1)
        self.layout.addWidget(self.sinPowerEdit, 6, 12, 1, 1)
        self.layout.addWidget(self.thetaStepLabel, 7, 11, 1, 1)
        self.layout.addWidget(self.thetaStepEdit, 7, 12, 1, 1)
        self.layout.addWidget(self.deltaAngleLabel, 8, 11, 1, 1)
        self.layout.addWidget(self.deltaAngleEdit, 8, 12, 1, 1)

#        self.layout.addWidget(self.FFT2Button, 6, 11, 1, 2)
#        self.layout.addWidget(self.fftThrEdit, 7, 11, 1, 1)

#        self.layout.addWidget(self.pointsButton, 8, 11, 1, 2)
#        self.layout.addWidget(self.pointsThrEdit, 9, 11, 1, 1)

        self.layout.addWidget(self.dirButton, 9, 11, 1, 2)
        self.layout.addWidget(self.lineLengthLabel, 10, 11, 1, 1)
        self.layout.addWidget(self.lineLengthEdit, 10, 12, 1, 1)
        self.layout.addWidget(self.sigmaLabel, 11, 11, 1, 1)
        self.layout.addWidget(self.sigmaEdit, 11, 12, 1, 1)
#
        self.layout.addWidget(self.filterImageButton, 12, 11, 1, 2)
        self.layout.addWidget(self.intThrButton, 13, 11, 1, 2)
        self.layout.addWidget(self.intThrLabel, 14, 11, 1, 1)
        self.layout.addWidget(self.intThrEdit, 14, 12, 1, 1)

        self.layout.addWidget(self.stormCropButton, 15, 11, 1, 2)

        self.roiSizeEdit.textChanged.connect(self.imagegui.updateROI)
        self.loadimageButton.clicked.connect(self.imagegui.loadImage)
        self.FFT2Button.clicked.connect(self.imagegui.FFT2)
        self.corrButton.clicked.connect(self.imagegui.corr2)
        self.pointsButton.clicked.connect(self.imagegui.points)
        self.filterImageButton.clicked.connect(self.imagegui.imageFilter)
        self.dirButton.clicked.connect(self.imagegui.getDirection)
        self.intThrButton.clicked.connect(self.imagegui.intThreshold)
        self.stormCropButton.clicked.connect(self.imagegui.stormCrop)


class ImageGUI(pg.GraphicsLayoutWidget):

    def __init__(self, main, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.main = main
        self.setWindowTitle('ImageGUI')
        self.subimgSize = 100

        # Item for displaying 10x10 image data
        self.vb1 = self.addViewBox(row=0, col=0)
        self.bigimg = pg.ImageItem()
        self.vb1.addItem(self.bigimg)
        self.vb1.setAspectLocked(True)

        # Custom ROI for selecting an image region
        self.roi = pg.ROI([0, 0], [0, 0])
        self.roiSizeX = np.float(self.main.roiSizeEdit.text())
        self.roiSizeY = np.float(self.main.roiSizeEdit.text())
        self.roi.setSize(self.roiSizeX, self.roiSizeY)
        self.roi.addScaleHandle([1, 1], [0, 0], lockAspect=True)
        self.vb1.addItem(self.roi)
        self.roi.setZValue(10)  # make sure ROI is drawn above image

        # Contrast/color control
        self.bigimg_hist = pg.HistogramLUTItem()
        self.bigimg_hist.gradient.loadPreset('thermal')
        self.bigimg_hist.setImageItem(self.bigimg)
        self.addItem(self.bigimg_hist, row=0, col=1)

        # subimg
        self.subimg = pg.ImageItem()
        subimg_hist = pg.HistogramLUTItem(image=self.subimg)
        subimg_hist.gradient.loadPreset('thermal')
        self.addItem(subimg_hist, row=1, col=1)

        self.dirPlot = pg.PlotItem(title="Subimage")
        self.dirPlot.addItem(self.subimg)
        self.addItem(self.dirPlot, row=1, col=0)

        # load image
        self.im = Image.open(r'/Users/Luciano/Documents/LabNanofisica/labnanofisica/ringfinder/spectrin1.tif')
        self.data = np.array(self.im)
        self.bigimg.setImage(self.data)
        self.bigimg_hist.setLevels(self.data.min(), self.data.max())

        # grid, n = number of divisions in each side
        self.setGrid(np.int(np.shape(self.data)[0]/50))

#        # FFT2
#        self.FFT2img = pg.ImageItem()
#        self.FFT2img_hist = pg.HistogramLUTItem(image=self.FFT2img)
#        self.FFT2img_hist.gradient.loadPreset('thermal')
#        self.addItem(self.FFT2img_hist, row=1, col=3)
#
#        self.fft2 = pg.PlotItem(title="FFT 2D")
#        self.fft2.addItem(self.FFT2img)
#        self.addItem(self.fft2,row=1,col=2)

#        # Points
#        self.pointsImg = pg.ImageItem()
#        self.pointsImg_hist = pg.HistogramLUTItem(image=self.pointsImg)
#        self.pointsImg_hist.gradient.loadPreset('thermal')
#        self.addItem(self.pointsImg_hist, row=1, col=5)
#
#        self.pointsPlot = pg.PlotItem(title="Points")
#        self.pointsPlot.addItem(self.pointsImg)
#        self.addItem(self.pointsPlot,row=1,col=4)

        # Correlation
        self.pCorr = pg.PlotItem(title="Correlation")
        self.addItem(self.pCorr, row=0, col=2)

        # Optimal correlation visualization
        self.vb4 = self.addViewBox(row=1, col=2)
        self.img1 = pg.ImageItem()
        self.img2 = pg.ImageItem()
        self.vb4.addItem(self.img1)
        self.vb4.addItem(self.img2)
        self.img2.setZValue(10)  # make sure this image is on top
        self.img2.setOpacity(0.5)
        self.vb4.setAspectLocked(True)
        overlay_hist = pg.HistogramLUTItem()
        overlay_hist.gradient.loadPreset('thermal')
        overlay_hist.setImageItem(self.img2)
        overlay_hist.setImageItem(self.img1)
        self.addItem(overlay_hist, row=1, col=3)

        self.roi.sigRegionChanged.connect(self.updatePlot)

    def loadImage(self):

        self.vb1.clear()
        pxSize = np.float(self.main.pxSizeEdit.text())
        subimgPxSize = 1000/pxSize
        self.filename = getFilename("Load image", [('Tiff file', '.tif')])
        self.loadedImage = Image.open(self.filename)
        self.data = np.array(self.loadedImage)
        self.bigimg = pg.ImageItem()
        self.vb1.addItem(self.bigimg)
        self.vb1.setAspectLocked(True)
        self.bigimg.setImage(self.data)
        self.bigimg_hist.setImageItem(self.bigimg)
        self.addItem(self.bigimg_hist, row=0, col=1)
        self.vb1.addItem(self.roi)
        self.setGrid(np.int(np.shape(self.data)[0]/subimgPxSize))

    # Callbacks for handling user interaction

    def updatePlot(self):

        self.dirPlot.clear()
        self.selected = self.roi.getArrayRegion(self.data, self.bigimg)
        self.subimg.setImage(self.selected)
        self.subimgSize = np.shape(self.selected)[0]
        self.dirPlot.addItem(self.subimg)

    def updateROI(self):

        self.roiSizeX = np.float(self.main.roiSizeEdit.text())
        self.roiSizeY = np.float(self.main.roiSizeEdit.text())
        self.roi.setSize(self.roiSizeX, self.roiSizeY)

    def FFT2(self):

        print('FFT 2D analysis executed')

        # remove previous fft2
        self.fft2.clear()

        # calculate new fft2
        fft2output = np.real(np.fft.fftshift(np.fft.fft2(self.selected)))

        # take abs value and log10 for better visualization
        fft2output = np.abs(np.log10(fft2output))

        # add new fft2 graph
        self.fft2.addItem(self.FFT2img)
        self.fft2.setAspectLocked(True)
        self.FFT2img.setImage(fft2output)

        # set fft2 analysis threshold
        self.fftThr = np.float(self.main.fftThrEdit.text())

        # calculate local intensity maxima
        coord = peak_local_max(fft2output, min_distance=2,
                               threshold_rel=self.fftThr)

        # take first 3 max
        coord = firstNmax(coord, fft2output, N=3)

        # size of the subimqge of interest
        A = self.subimgSize

        # max and min radius in pixels, 9 -> 220 nm, 12 -> 167 nm
        rmin = 9
        rmax = 12

        # draw circles for visulization
        rminX = A*(rmax/100)*np.cos(np.linspace(0, 2*np.pi, 1000))+A/2
        rminY = A*(rmax/100)*np.sin(np.linspace(0, 2*np.pi, 1000))+A/2
        rmaxX = A*(rmin/100)*np.cos(np.linspace(0, 2*np.pi, 1000))+A/2
        rmaxY = A*(rmin/100)*np.sin(np.linspace(0, 2*np.pi, 1000))+A/2
        self.fft2.plot(rminX, rminY, pen=(0, 102, 204))
        self.fft2.plot(rmaxX, rmaxY, pen=(0, 102, 204))

        # plot local maxima
        self.fft2.plot(coord[:, 0], coord[:, 1], pen=None,
                       symbolBrush=(0, 102, 204), symbolPen='w')
#        self.fft2.plot(maxfinder.positions[:, 0], maxfinder.positions[:, 1],
#                       pen=None, symbolBrush=(204,102,0), symbolPen='w')

        # aux arrays: ringBool is checked to define if there are rings or not
        ringBool = []

        # D saves the distances of local maxima from the centre of the fft2
        D = []

        # loop for calculating all the distances d, elements of array D

        for i in np.arange(0, np.shape(coord)[0]):
            d = dist([A/2, A/2], coord[i])
            D.append(dist([A/2, A/2], coord[i]))
            if A*(rmin/100) < d < A*(rmax/100):
                ringBool.append(1)

        # condition for ringBool: all elements d must correspond to
        # periods between 170 and 220 nm
        if np.sum(ringBool) == np.shape(coord)[0]-1 and np.sum(ringBool) > 0:
            print('¡HAY ANILLOS!')
        else:
            print('NO HAY ANILLOS')

    def points(self):

        print('Points analysis executed')

        # clear previous plot
        self.pointsPlot.clear()

        # set points analysis thereshold
        self.pointsThr = np.float(self.main.pointsThrEdit.text())

        # find local maxima (points)
        points = peak_local_max(self.selected, min_distance=3,
                                threshold_rel=self.pointsThr)

        # take first N=7 points
        points = firstNmax(points, self.selected, N=7)

        # plot points
        self.pointsPlot.addItem(self.pointsImg)
        self.pointsImg.setImage(self.selected)
        self.pointsPlot.plot(points[:, 0], points[:, 1], pen=None,
                             symbolBrush=(0, 204, 122), symbolPen='w')

        # set minimum and maximum distance between points
        dmin = 8
        dmax = 12

        # instance of D, vector with groups of three points
        D = []

        # look up every point
        for i in np.arange(0, np.shape(points)[0]-1):
            # calculate the distance of every point to the others
            for j in np.arange(i+1, np.shape(points)[0]):
                d1 = dist(points[i], points[j])
                # if there are two points at the right distance then
                if dmin < d1 < dmax:
                    for k in np.arange(0, np.shape(points)[0]-1):
                        # check the distance between the last point
                        # and the other points in the list
                        if k != i and k != j:
                            d2 = dist(points[j], points[k])

                        else:
                            d2 = 0

                        # calculate the angle between vector i-j and j-k with i, j, k points
                        v1 = points[i]-points[j]
                        v2 = points[j]-points[k]
                        t = cosTheta(v1, v2)

                        # if point k is at right distance from point j and the angle is flat enough
                        if dmin < d2 < dmax and np.abs(t) > 0.8 :
                            # save the three points
                            D.append([points[i], points[j], points[k]])
                            # plot connections
                            pen = pg.mkPen(color=(0, 255, 100), width=1, 
                                           style=QtCore.Qt.SolidLine, 
                                           antialias=True)
                            self.pointsPlot.plot([points[i][0], points[j][0], 
                                                 points[k][0]], [points[i][1],
                                                 points[j][1], points[k][1]],
                                                 pen=pen,
                                                 symbolBrush=(0, 204, 122),
                                                 symbolPen='w')
                        else:
                            pass
                else:
                    pass

        if len(D) > 0:
            print('¡HAY ANILLOS!')
        else:
            print('NO HAY ANILLOS')


#        print(D)
#        print(len(D))
#        print(np.shape(D))

    def corr2(self):

        corr2thr = np.float(self.main.corr2thrEdit.text())
        self.pCorr.clear()

        n = np.float(self.main.thetaStepEdit.text())
        theta = np.arange(0, 180, n)

        self.thetaSteps = np.arange(0, (180/n), 1)
        self.phaseSteps = np.arange(0, 21, 1)

        self.R = np.zeros(np.size(self.thetaSteps))
        self.R2 = np.zeros(np.size(self.phaseSteps))
        R22 = np.zeros(np.size(self.thetaSteps))
        R22ph = np.zeros(np.size(self.thetaSteps))

        wvlen_nm = np.float(self.main.wvlenEdit.text())  # wvlen in nm
        pxSize = np.float(self.main.pxSizeEdit.text())
        wvlen = wvlen_nm/pxSize  # wvlen in px
        sinPower = np.float(self.main.sinPowerEdit.text())

        # now we correlate with the full sin2D pattern
        for i in self.thetaSteps:
            for p in self.phaseSteps:
                self.axonTheta = simAxon(imSize=self.subimgSize, wvlen=wvlen,
                                         theta=n*i, phase=p*.025, a=0,
                                         b=sinPower).simAxon
                r = corr2(self.selected, self.axonTheta)
                self.R2[p] = r
            R22[i-1] = np.max(self.R2)
            # R22ph saves the phase that maximizes correlation
            R22ph[i-1] = .025*np.argmax(self.R2)

        thetaMax = theta[np.argmax(R22)]
        phaseMax = R22ph[np.argmax(R22)]

        self.bestAxon = simAxon(imSize=self.subimgSize, wvlen=wvlen,
                                theta=thetaMax, phase=phaseMax,
                                a=0, b=sinPower).simAxon

        self.img1.setImage(self.bestAxon)
        self.img2.setImage(self.selected)

        pen1 = pg.mkPen(color=(0, 255, 100), width=2,
                        style=QtCore.Qt.SolidLine, antialias=True)
        pen2 = pg.mkPen(color=(255, 50, 60), width=1,
                        style=QtCore.Qt.SolidLine, antialias=True)
        self.pCorr.plot(theta, R22, pen=pen1)
        self.pCorr.plot(theta, corr2thr*np.ones(np.size(theta)), pen=pen2)
        self.pCorr.showGrid(x=False, y=False)

        if self.meanAngle is not None:
            angleMax = np.float(self.main.deltaAngleEdit.text())
            deltaAngle = np.arange(self.meanAngle-angleMax,
                                   self.meanAngle+angleMax, dtype=int)
            self.pCorr.plot(deltaAngle, 0.3*np.ones(np.size(deltaAngle)),
                            fillLevel=0, brush=(50, 50, 200, 100))
        else:
            pass

        self.pCorr.showGrid(x=True, y=True)

        # extension to deal with angles close to 0 or 180
        R22 = arrayExt(R22)
        deltaAngle = np.arange(180+self.meanAngle-angleMax,
                               180+self.meanAngle+angleMax, dtype=int)

        if np.max(R22[np.array(deltaAngle/n, dtype=int)]) > corr2thr:
            print('¡HAY ANILLOS!')
        else:
            print('NO HAY ANILLOS')

    def intThreshold(self):
        # TO DO: find better criteria
        thr = np.float(self.main.intThrEdit.text())
        intMean = np.mean(self.data)
        if np.mean(self.selected) < intMean/thr:
            print('BACKGROUND')
        else:
            print('NEURON')

    def imageFilter(self):

        sigma = np.float(self.main.sigmaEdit.text())
        img = ndi.gaussian_filter(self.data, sigma)

        thresh = threshold_otsu(img)
        binary = img > thresh

        self.data = self.data*binary
        self.bigimg.setImage(self.data)

    def getDirection(self):

        # gaussian filter to get low resolution image
        sigma = np.float(self.main.sigmaEdit.text())
        img = ndi.gaussian_filter(self.selected, sigma)

        # binarization of image
        thresh = threshold_otsu(img)
        binary = img > thresh

        # find edges
        edges = filters.sobel(binary)

        print(self.subimgSize*(2/5))
        # get directions
        linLen = np.float(self.main.lineLengthEdit.text())
        lines = probabilistic_hough_line(edges, threshold=10,
                                         line_length=self.subimgSize*linLen,
                                         line_gap=3)

        # plot and save the angles of the lines
        angleArr = []
        for line in lines:
            p0, p1 = line
            pen = pg.mkPen(color=(0, 255, 100), width=1,
                           style=QtCore.Qt.SolidLine, antialias=True)
            self.dirPlot.plot((p0[1], p1[1]), (p0[0], p1[0]), pen=pen)

            # get the m coefficient of the lines and the angle
            m = (p1[0]-p0[0])/(p1[1]-p0[1])
            angle = (180/np.pi)*np.arctan(m)

            if angle < 0:
                angle = angle + 180
            else:
                pass

            angleArr.append(angle)

        # calculate mean angle and its standard deviation
        print(angleArr)
        self.meanAngle = np.mean(angleArr)
        self.stdAngle = np.std(angleArr)

        print('Angle is {} +/- {}'.format(self.meanAngle, self.stdAngle))

    def setGrid(self, n):

        pen = QtGui.QPen(QtCore.Qt.yellow, 1, QtCore.Qt.SolidLine)

        self.xlines = []
        self.ylines = []

        for i in np.arange(0, n-1):
            self.xlines.append(pg.InfiniteLine(pen=pen, angle=0))
            self.ylines.append(pg.InfiniteLine(pen=pen))

        for i in np.arange(0, n-1):
            self.xlines[i].setPos((np.shape(self.data)[0]/n)*(i+1))
            self.ylines[i].setPos((np.shape(self.data)[0]/n)*(i+1))
            self.vb1.addItem(self.xlines[i])
            self.vb1.addItem(self.ylines[i])

    def stormCrop(self):

        self.data = self.data[59:-59, 59:-59]
        self.vb1.clear()
        self.vb1.addItem(self.bigimg)
        self.vb1.setAspectLocked(True)
        self.bigimg.setImage(self.data)
        self.bigimg_hist.setImageItem(self.bigimg)
        self.addItem(self.bigimg_hist, row=0, col=1)
        self.vb1.addItem(self.roi)
        pxSize = np.float(self.main.pxSizeEdit.text())
        subimgPxSize = 1000/pxSize
        self.setGrid(np.int(np.shape(self.data)[0]/subimgPxSize))

"""
Tormenta code for finding maxima
"""


## data-type definitions
#def fit_par(fit_model):
#    if fit_model is '2d':
#        return [('amplitude_fit', float), ('fit_x', float), ('fit_y', float),
#                ('background_fit', float)]
#
#
#def results_dt(fit_parameters):
#    parameters = [('frame', int), ('maxima_x', int), ('maxima_y', int),
#                  ('photons', float), ('sharpness', float),
#                  ('roundness', float), ('brightness', float)]
#    return np.dtype(parameters + fit_parameters)
#
#
#class Maxima():
#    """ Class defined as the local maxima in an image frame. """
#
#    def __init__(self, image, fit_par=None, dt=0, fw=None, win_size=None,
#                 kernel=None, xkernel=None, bkg_image=None):
#
#        self.image = image
#        self.bkg_image = bkg_image
#
#        # Noise removal by convolving with a null sum gaussian. Its FWHM
#        # has to match the one of the objects we want to detect.
#        try:
#            self.fwhm = fw
#            self.win_size = win_size
#            self.kernel = kernel
#            self.xkernel = xkernel
#            self.image_conv = ndi.filters.convolve(self.image.astype(float), self.kernel)
#        except RuntimeError:
#            # If the kernel is None, I assume all the args must be calculated
#            self.fwhm = gaussians.get_fwhm(670, 1.42) / 120
#            self.win_size = int(np.ceil(self.fwhm))
#            self.kernel = gaussians.kernel(self.fwhm)
#            self.xkernel = gaussians.xkernel(self.fwhm)
#            self.image_conv = ndi.filters.convolve(self.image.astype(float), self.kernel)
#
#        # TODO: FIXME
#        if self.bkg_image is None:
#            self.bkg_image = self.image_conv
#
#        self.fit_par = fit_par
#        self.dt = dt
#
#    def find_old(self, alpha=5):
#        """Local maxima finding routine.
#        Alpha is the amount of standard deviations used as a threshold of the
#        local maxima search. Size is the semiwidth of the fitting window.
#        Adapted from http://stackoverflow.com/questions/16842823/
#                            peak-detection-in-a-noisy-2d-array
#        """
#        self.alpha = alpha
#
#        # Image mask
#        self.imageMask = np.zeros(self.image.shape, dtype=bool)
#
#        self.mean = np.mean(self.image_conv)
#        self.std = np.sqrt(np.mean((self.image_conv - self.mean)**2))
#        self.threshold = self.alpha*self.std + self.mean
#
#        # Estimate for the maximum number of maxima in a frame
#        nMax = self.image.size // (2*self.win_size + 1)**2
#        self.positions = np.zeros((nMax, 2), dtype=int)
#        nPeak = 0
#
#        while 1:
#            k = np.argmax(np.ma.masked_array(self.image_conv, self.imageMask))
#
#            # index juggling
#            j, i = np.unravel_index(k, self.image.shape)
#            if(self.image_conv[j, i] >= self.threshold):
#
#                # Saving the peak
#                self.positions[nPeak] = tuple([j, i])
#
#                # this is the part that masks already-found maxima
#                x = np.arange(i - self.win_size, i + self.win_size + 1,
#                              dtype=np.int)
#                y = np.arange(j - self.win_size, j + self.win_size + 1,
#                              dtype=np.int)
#                xv, yv = np.meshgrid(x, y)
#                # the clip handles cases where the peak is near the image edge
#                self.imageMask[yv.clip(0, self.image.shape[0] - 1),
#                               xv.clip(0, self.image.shape[1] - 1)] = True
#                nPeak += 1
#            else:
#                break
#
#        if nPeak > 0:
#            self.positions = self.positions[:nPeak]
#            self.drop_overlapping()
#            self.drop_border()
#
#    def find(self, alpha=5):
#        """
#        Takes an image and detect the peaks usingthe local maximum filter.
#        Returns a boolean mask of the peaks (i.e. 1 when
#        the pixel's value is the neighborhood maximum, 0 otherwise). Taken from
#        http://stackoverflow.com/questions/9111711/
#        get-coordinates-of-local-maxima-in-2d-array-above-certain-value
#        """
#        self.alpha = alpha
#
#        image_max = ndi.filters.maximum_filter(self.image_conv, self.win_size)
#        maxima = (self.image_conv == image_max)
#
#        self.mean = np.mean(self.image_conv)
#        self.std = np.sqrt(np.mean((self.image_conv - self.mean)**2))
#        self.threshold = self.alpha*self.std + self.mean
#
#        diff = (image_max > self.threshold)
#        maxima[diff == 0] = 0
#
#        labeled, num_objects = ndi.label(maxima)
#        if num_objects > 0:
#            self.positions = ndi.measurements.maximum_position(self.image, labeled,
#                                              range(1, num_objects + 1))
#            self.positions = np.array(self.positions).astype(int)
#            self.drop_overlapping()
#            self.drop_border()
#        else:
#            self.positions = np.zeros((0, 2), dtype=int)
#
#    def drop_overlapping(self):
#        """Drop overlapping spots."""
#        n = len(self.positions)
#        if n > 1:
#            self.positions = tools.dropOverlapping(self.positions,
#                                                   2*self.win_size + 1)
#            self.overlaps = n - len(self.positions)
#        else:
#            self.overlaps = 0
#
#    def drop_border(self):
#        """ Drop near-the-edge spots. """
#        ws = self.win_size
#        lx = self.image.shape[0] - ws
#        ly = self.image.shape[1] - ws
#        keep = ((self.positions[:, 0] < lx) & (self.positions[:, 0] > ws) &
#                (self.positions[:, 1] < ly) & (self.positions[:, 1] > ws))
#        self.positions = self.positions[keep]
#
#    def getParameters(self):
#        """Calculate the roundness, brightness, sharpness"""
#
#        # Results storage
#        try:
#            self.results = np.zeros(len(self.positions), dtype=self.dt)
#        except TypeError:
#            self.fit_model = '2d'
#            self.fit_par = fit_par(self.fit_model)
#            self.dt = results_dt(self.fit_par)
#            self.results = np.zeros(len(self.positions), dtype=self.dt)
#
#        self.results['maxima_x'] = self.positions[:, 0]
#        self.results['maxima_y'] = self.positions[:, 1]
#
#        mask = np.zeros((2*self.win_size + 1, 2*self.win_size + 1), dtype=bool)
#        mask[self.win_size, self.win_size] = True
#
#        i = 0
#        for maxx in self.positions:
#            # tuples make indexing easier (see below)
#            p = tuple(maxx)
#            masked = np.ma.masked_array(self.radius(self.image, maxx), mask)
#
#            # Sharpness
#            sharp_norm = self.image_conv[p] * np.mean(masked)
#            self.results['sharpness'][i] = 100*self.image[p]/sharp_norm
#            # Roundness
#            hx = np.dot(self.radius(self.image, maxx)[2, :], self.xkernel)
#            hy = np.dot(self.radius(self.image, maxx)[:, 2], self.xkernel)
#            self.results['roundness'][i] = 2 * (hy - hx) / (hy + hx)
#            # Brightness
#            bright_norm = self.alpha * self.std
#            self.results['brightness'][i] = 2.5*np.log(self.image_conv[p] /
#                                                       bright_norm)
#
#            i += 1
#
#    def area(self, image, n):
#        """Returns the area around the local maximum number n."""
#        coord = self.positions[n]
#        x1 = coord[0] - self.win_size
#        x2 = coord[0] + self.win_size + 1
#        y1 = coord[1] - self.win_size
#        y2 = coord[1] + self.win_size + 1
#        return image[x1:x2, y1:y2]
#
#    def radius(self, image, coord):
#        """Returns the area around the entered point."""
#        x1 = coord[0] - self.win_size
#        x2 = coord[0] + self.win_size + 1
#        y1 = coord[1] - self.win_size
#        y2 = coord[1] + self.win_size + 1
#        return image[x1:x2, y1:y2]
#
#    def fit(self, fit_model='2d'):
#
#        self.mean_psf = np.zeros(self.area(self.image, 0).shape)
#
#        for i in np.arange(len(self.positions)):
#
#            # Fit and store fitting results
#            area = self.area(self.image, i)
#            bkg = self.area(self.bkg_image, i)
#            fit = fit_area(area, self.fwhm, bkg)
#            offset = self.positions[i] - self.win_size
#            fit[1] += offset[0]
#            fit[2] += offset[1]
#
#            # Can I do this faster if fit_area returned a struct array?
#            m = 0
#            for par in self.fit_par:
#                self.results[par[0]][i] = fit[m]
#                m += 1
#
#            # Background-sustracted measured PSF
#            bkg_subtract = area - fit[-1]
#            # photons from molecule calculation
#            self.results['photons'][i] = np.sum(bkg_subtract)
#            self.mean_psf += bkg_subtract / self.results['photons'][i]

if __name__ == '__main__':

    app = QtGui.QApplication([])

    win = RingAnalizer()
    win.show()
    app.exec_()
