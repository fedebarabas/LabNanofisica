# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 13:41:48 2014

@authors: Luciano Masullo, Federico Barabas
"""

import os
import numpy as np

from scipy import ndimage as ndi
import skimage.filter as filters
from skimage.transform import probabilistic_hough_line

from PIL import Image
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

import labnanofisica.utils as utils
from labnanofisica.ringfinder.neurosimulations import simAxon
import labnanofisica.ringfinder.tools as tools


class GollumDeveloper(QtGui.QMainWindow):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.setWindowTitle('Ring Finder Developer')

        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)

        self.setFixedSize(1275, 800)

        self.FFT2Button = QtGui.QPushButton('FFT 2D')
        self.corrButton = QtGui.QPushButton('Correlation')
        self.corrThresEdit = QtGui.QLineEdit('0.1')
        self.thetaStepEdit = QtGui.QLineEdit('3')
        self.deltaThEdit = QtGui.QLineEdit('30')
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
        self.corrThresLabel = QtGui.QLabel('Correlation threshold')
        self.thetaStepLabel = QtGui.QLabel('Angular step (°)')
        self.deltaThLabel = QtGui.QLabel('Delta Angle (°)')
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
        self.layout.addWidget(self.corrThresLabel, 4, 11, 1, 1)
        self.layout.addWidget(self.corrThresEdit, 4, 12, 1, 1)
        self.layout.addWidget(self.wvlenLabel, 5, 11, 1, 1)
        self.layout.addWidget(self.wvlenEdit, 5, 12, 1, 1)
        self.layout.addWidget(self.sinPowerLabel, 6, 11, 1, 1)
        self.layout.addWidget(self.sinPowerEdit, 6, 12, 1, 1)
        self.layout.addWidget(self.thetaStepLabel, 7, 11, 1, 1)
        self.layout.addWidget(self.thetaStepEdit, 7, 12, 1, 1)
        self.layout.addWidget(self.deltaThLabel, 8, 11, 1, 1)
        self.layout.addWidget(self.deltaThEdit, 8, 12, 1, 1)

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
#        self.FFT2Button.clicked.connect(self.imagegui.FFT2)
#        self.corrButton.clicked.connect(self.imagegui.corr2)
#        self.pointsButton.clicked.connect(self.imagegui.points)
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
        path = os.path.join(os.getcwd(), 'labnanofisica', 'ringfinder',
                            'spectrin1.tif')
        self.im = Image.open(path)
        self.data = np.array(self.im)
        self.bigimg.setImage(self.data)
        self.bigimg_hist.setLevels(self.data.min(), self.data.max())

        # grid, n = number of divisions in each side
        self.setGrid(self.vb1, self.data, np.int(np.shape(self.data)[0]/50))

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
        self.filename = utils.getFilename("Load image",
                                          [('Tiff file', '.tif')])
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

    def FFTMethodGUI(self):

        print('FFT 2D analysis executed')

        thres = np.float(self.main.fftThrEdit.text())
        fft2output, coord, rlim, rings = tools.FFTMethod(self.selected, thres)
        rmin, rmax = rlim

        if rings:
            print('¡HAY ANILLOS!')
        else:
            print('NO HAY ANILLOS')

        # plot results
        self.fft2.clear()       # remove previous fft2
        self.fft2.addItem(self.FFT2img)
        self.fft2.setAspectLocked(True)
        self.FFT2img.setImage(fft2output)

        A = self.subimgSize    # size of the subimqge of interest

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

    def pointsMethodGUI(self):

        print('Points analysis executed')

        # clear previous plot
        self.pointsPlot.clear()

        # set points analysis thereshold
        thres = np.float(self.main.pointsThrEdit.text())
        points, D, rings = tools.pointsMethod(self.selected, thres)

        if rings:
            print('¡HAY ANILLOS!')
            pen = pg.mkPen(color=(0, 255, 100), width=1,
                           style=QtCore.Qt.SolidLine, antialias=True)
            for d in D:
                self.pointsPlot.plot([d[0][0], d[1][0], d[2][0]],
                                     [d[0][1], d[1][1], d[2][1]], pen=pen,
                                     symbolBrush=(0, 204, 122), symbolPen='w')
        else:
            print('NO HAY ANILLOS')

        # plot results
        self.pointsPlot.addItem(self.pointsImg)
        self.pointsImg.setImage(self.selected)
        self.pointsPlot.plot(points[:, 0], points[:, 1], pen=None,
                             symbolBrush=(0, 204, 122), symbolPen='w')

    def corrMethodGUI(self):

        self.pCorr.clear()

        # for every subimg, we apply the correlation method for
        # ring finding
        corrThres = np.float(self.corrThresEdit.text())
        thStep = np.float(self.thetaStepEdit.text())
        deltaTh = np.float(self.deltaThEdit.text())
        wvlen = np.float(self.wvlenEdit.text())
        sinPow = np.float(self.sinPowerEdit.text())
        args = [corrThres, np.float(self.sigmaEdit.text()),
                np.float(self.pxSizeEdit.text()),
                np.float(self.lineLengthEdit.text()),
                thStep, deltaTh, wvlen, sinPow]
        output = tools.corrMethod(self.selected, *args)
        th0, corrTheta, corrMax, thetaMax, phaseMax, rings = output

        self.bestAxon = simAxon(imSize=self.subimgSize, wvlen=wvlen,
                                theta=thetaMax, phase=phaseMax, b=sinPow).data

        self.img1.setImage(self.bestAxon)
        self.img2.setImage(self.selected)

        # plot the threshold of correlation chosen by the user
        # phase steps are set to 20, TO DO: explore this parameter
        theta = np.arange(0, 180, thStep)
        pen1 = pg.mkPen(color=(0, 255, 100), width=2,
                        style=QtCore.Qt.SolidLine, antialias=True)
        pen2 = pg.mkPen(color=(255, 50, 60), width=1,
                        style=QtCore.Qt.SolidLine, antialias=True)
        self.pCorr.plot(theta, corrTheta, pen=pen1)
        self.pCorr.plot(theta, corrThres*np.ones(np.size(theta)), pen=pen2)
        self.pCorr.showGrid(x=False, y=False)

        # plot the area given by the direction (meanAngle) and the deltaTh
        if self.meanAngle is not None:
            thetaArea = np.arange(self.meanAngle - deltaTh,
                                  self.meanAngle + deltaTh, dtype=int)
            self.pCorr.plot(thetaArea, 0.3*np.ones(np.size(deltaTh)),
                            fillLevel=0, brush=(50, 50, 200, 100))
        self.pCorr.showGrid(x=True, y=True)

        if rings:
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

        thresh = filters.threshold_otsu(img)
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
        thresh = filters.threshold_otsu(img)
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
        tools.setGrid(self.vb1, self.data,
                      np.int(np.shape(self.data)[0]/subimgPxSize))

if __name__ == '__main__':

    app = QtGui.QApplication([])

    win = GollumDeveloper()
    win.show()
    app.exec_()
