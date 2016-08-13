# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 13:41:48 2014

@authors: Luciano Masullo
"""

from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import filters
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
import numpy as np
from PIL import Image
from labnanofisica.ringfinder.neurosimulations import simAxon
from skimage.filters import threshold_otsu, sobel

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from tkinter import Tk, filedialog, simpledialog



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

        self.setWindowTitle('Ring Finder Developer')

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
        self.sigmaEdit = QtGui.QLineEdit('60')
        self.intThrButton = QtGui.QPushButton('Intensity threshold')
        self.intThrEdit = QtGui.QLineEdit('3')
        self.lineLengthEdit = QtGui.QLineEdit('.4')
        self.wvlenEdit = QtGui.QLineEdit('180')

        self.roiLabel = QtGui.QLabel('ROI size (px)')
        self.corr2thrLabel = QtGui.QLabel('Correlation threshold')
        self.thetaStepLabel = QtGui.QLabel('Angular step (°)')
        self.deltaAngleLabel = QtGui.QLabel('Delta Angle (°)')
        self.sigmaLabel = QtGui.QLabel('Sigma of gaussian filter (nm)')
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
        self.im = Image.open(r'C:\Users\luciano.masullo\Documents\GitHub\LabNanofisica\labnanofisica\ringfinder\spectrin1.tif')
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

    def corr2(self):

        corr2thr = np.float(self.main.corr2thrEdit.text())
        self.pCorr.clear()

        n = np.float(self.main.thetaStepEdit.text())
        theta = np.arange(0, 180, n)

        thetaSteps = np.arange(0, (180/n), 1)
        phaseSteps = np.arange(0, 21, 1)

        corrPhase = np.zeros(np.size(phaseSteps))
        corrAngle = np.zeros(np.size(thetaSteps))
        corrPhaseArg = np.zeros(np.size(thetaSteps))

        wvlen_nm = np.float(self.main.wvlenEdit.text())  # wvlen in nm
        pxSize = np.float(self.main.pxSizeEdit.text())
        wvlen = wvlen_nm/pxSize  # wvlen in px
        sinPower = np.float(self.main.sinPowerEdit.text())

        # now we correlate with the full sin2D pattern
        for i in thetaSteps:
            for p in phaseSteps:
                self.axonTheta = simAxon(imSize=self.subimgSize, wvlen=wvlen,
                                         theta=n*i, phase=p*.025, a=0,
                                         b=sinPower).simAxon
                r = corr2(self.selected, self.axonTheta)
                corrPhase[p] = r
            corrAngle[i-1] = np.max(corrPhase)
            # corrPhaseArg saves the phase that maximizes correlation
            corrPhaseArg[i-1] = .025*np.argmax(corrPhase)

        # plot best correlation with the image
        thetaMax = theta[np.argmax(corrAngle)]
        phaseMax = corrPhaseArg[np.argmax(corrAngle)]

        self.bestAxon = simAxon(imSize=self.subimgSize, wvlen=wvlen,
                                theta=thetaMax, phase=phaseMax,
                                a=0, b=sinPower).simAxon

        self.img1.setImage(self.bestAxon)
        self.img2.setImage(self.selected)

        pen1 = pg.mkPen(color=(0, 255, 100), width=2,
                        style=QtCore.Qt.SolidLine, antialias=True)
        pen2 = pg.mkPen(color=(255, 50, 60), width=1,
                        style=QtCore.Qt.SolidLine, antialias=True)

        # plot the threshold of correlation chosen by the user
        self.pCorr.plot(theta, corrAngle, pen=pen1)
        self.pCorr.plot(theta, corr2thr*np.ones(np.size(theta)), pen=pen2)
        self.pCorr.showGrid(x=False, y=False)

        # plot the area given by the direction (meanAngle) and the deltaAngle
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
        corrAngle = arrayExt(corrAngle)
        deltaAngle = np.arange(180+self.meanAngle-angleMax,
                               180+self.meanAngle+angleMax, dtype=int)

        # decide wether there are rings or not
        if np.max(corrAngle[np.array(deltaAngle/n, dtype=int)]) > corr2thr:
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
        pxSize = np.float(self.main.pxSizeEdit.text())
        sigma_px = sigma/pxSize
        img = ndi.gaussian_filter(self.selected, sigma_px)

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

if __name__ == '__main__':

    app = QtGui.QApplication([])

    win = RingAnalizer()
    win.show()
    app.exec_()
