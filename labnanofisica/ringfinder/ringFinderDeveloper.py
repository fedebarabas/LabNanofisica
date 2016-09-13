# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 13:41:48 2014

@authors: Luciano Masullo, Federico Barabas
"""

import os
import numpy as np

from scipy import ndimage as ndi
import skimage.filters as filters

from PIL import Image
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

import labnanofisica.utils as utils
from labnanofisica.ringfinder.neurosimulations import simAxon
import labnanofisica.ringfinder.tools as tools


class GollumDeveloper(QtGui.QMainWindow):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.setWindowTitle('Gollum Developer')

        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)

        self.subImgSize = 1000      # subimage size in nm

        # Separate frame for loading controls
        loadFrame = QtGui.QFrame(self)
        loadFrame.setFrameStyle(QtGui.QFrame.Panel)
        loadLayout = QtGui.QGridLayout()
        loadFrame.setLayout(loadLayout)
        loadTitle = QtGui.QLabel('<strong>Load image</strong>')
        loadTitle.setTextFormat(QtCore.Qt.RichText)
        loadLayout.addWidget(loadTitle, 0, 0)
        loadLayout.addWidget(QtGui.QLabel('STORM pixel [nm]'), 1, 0)
        self.STORMPxEdit = QtGui.QLineEdit('6.65')
        loadLayout.addWidget(self.STORMPxEdit, 1, 1)
        excludedLabel = QtGui.QLabel('#Excluded px from localization')
        loadLayout.addWidget(excludedLabel, 2, 0)
        self.excludedEdit = QtGui.QLineEdit('3')
        loadLayout.addWidget(self.excludedEdit, 2, 1)
        loadLayout.addWidget(QtGui.QLabel('STORM magnification'), 3, 0)
        self.magnificationEdit = QtGui.QLineEdit('20')
        loadLayout.addWidget(self.magnificationEdit, 3, 1)
        self.loadSTORMButton = QtGui.QPushButton('Load STORM Image')
        loadLayout.addWidget(self.loadSTORMButton, 4, 0, 1, 2)
        loadLayout.addWidget(QtGui.QLabel('STED pixel [nm]'), 5, 0)
        self.STEDPxEdit = QtGui.QLineEdit('20')
        loadLayout.addWidget(self.STEDPxEdit, 5, 1)
        self.loadSTEDButton = QtGui.QPushButton('Load STED Image')
        loadLayout.addWidget(self.loadSTEDButton, 6, 0, 1, 2)
        loadFrame.setFixedHeight(200)

        # Ring finding method settings
        self.corrThresEdit = QtGui.QLineEdit('0.07')
        self.thetaStepEdit = QtGui.QLineEdit('3')
        self.deltaThEdit = QtGui.QLineEdit('20')
        self.sinPowerEdit = QtGui.QLineEdit('3')
        self.loadimageButton = QtGui.QPushButton('Load Image')
        self.filterImageButton = QtGui.QPushButton('Filter Image')
        self.fftThrEdit = QtGui.QLineEdit('0.6')
        self.roiSizeEdit = QtGui.QLineEdit('1000')
        self.dirButton = QtGui.QPushButton('Get direction')
        self.sigmaEdit = QtGui.QLineEdit('60')
        self.intThrButton = QtGui.QPushButton('Intensity threshold')
        self.intThrEdit = QtGui.QLineEdit('3')
        self.lineLengthEdit = QtGui.QLineEdit('300')
        self.wvlenEdit = QtGui.QLineEdit('180')
        self.corrButton = QtGui.QPushButton('Correlation')
        self.resultLabel = QtGui.QLabel()
        self.resultLabel.setAlignment(QtCore.Qt.AlignCenter |
                                      QtCore.Qt.AlignVCenter)
        self.resultLabel.setTextFormat(QtCore.Qt.RichText)

        self.roiLabel = QtGui.QLabel('ROI size [nm]')
        self.corrThresLabel = QtGui.QLabel('Correlation threshold')
        self.thetaStepLabel = QtGui.QLabel('Angular step [°]')
        self.deltaThLabel = QtGui.QLabel('Delta Angle [°]')
        self.sigmaLabel = QtGui.QLabel('Sigma of gaussian filter [nm]')
        self.sinPowerLabel = QtGui.QLabel('Sinusoidal pattern power')
        self.pxSizeLabel = QtGui.QLabel('Pixel size [nm]')
        self.intThrLabel = QtGui.QLabel('#sigmas threshold from mean')
        self.lineLengthLabel = QtGui.QLabel('Direction lines min length [nm]')
        # Direction lines lengths are expressed in fraction of subimg size
        self.wvlenLabel = QtGui.QLabel('wvlen of corr pattern [nm]')

        settingsFrame = QtGui.QFrame(self)
        settingsFrame.setFrameStyle(QtGui.QFrame.Panel)
        settingsLayout = QtGui.QGridLayout()
        settingsFrame.setLayout(settingsLayout)
        settingsTitle = QtGui.QLabel('<strong>Ring finding settings</strong>')
        settingsTitle.setTextFormat(QtCore.Qt.RichText)
        settingsLayout.addWidget(settingsTitle, 0, 0)
        settingsLayout.addWidget(self.roiLabel, 1, 0)
        settingsLayout.addWidget(self.roiSizeEdit, 1, 1)
        settingsLayout.addWidget(self.intThrLabel, 2, 0)
        settingsLayout.addWidget(self.intThrEdit, 2, 1)
        settingsLayout.addWidget(self.intThrButton, 3, 0, 1, 2)
        settingsLayout.addWidget(self.sigmaLabel, 4, 0)
        settingsLayout.addWidget(self.sigmaEdit, 4, 1)
        settingsLayout.addWidget(self.filterImageButton, 5, 0, 1, 2)
        settingsLayout.addWidget(self.lineLengthLabel, 6, 0)
        settingsLayout.addWidget(self.lineLengthEdit, 6, 1)
        settingsLayout.addWidget(self.dirButton, 7, 0, 1, 2)
        settingsLayout.addWidget(self.corrThresLabel, 8, 0)
        settingsLayout.addWidget(self.corrThresEdit, 8, 1)
        settingsLayout.addWidget(self.wvlenLabel, 9, 0)
        settingsLayout.addWidget(self.wvlenEdit, 9, 1)
        settingsLayout.addWidget(self.sinPowerLabel, 10, 0)
        settingsLayout.addWidget(self.sinPowerEdit, 10, 1)
        settingsLayout.addWidget(self.thetaStepLabel, 11, 0)
        settingsLayout.addWidget(self.thetaStepEdit, 11, 1)
        settingsLayout.addWidget(self.deltaThLabel, 12, 0)
        settingsLayout.addWidget(self.deltaThEdit, 12, 1)
        settingsLayout.addWidget(self.corrButton, 13, 0, 1, 2)
        settingsLayout.addWidget(self.resultLabel, 14, 0, 2, 0)
        settingsFrame.setFixedHeight(400)

        buttonWidget = QtGui.QWidget()
        buttonsLayout = QtGui.QGridLayout()
        buttonWidget.setLayout(buttonsLayout)
        buttonsLayout.addWidget(loadFrame, 0, 0)
        buttonsLayout.addWidget(settingsFrame, 1, 0)

        # Widgets' layout
        layout = QtGui.QGridLayout()
        self.cwidget.setLayout(layout)
        layout.addWidget(buttonWidget, 0, 0)
        self.imageWidget = ImageWidget(self)
        layout.addWidget(self.imageWidget, 0, 1)
        layout.setColumnMinimumWidth(1, 1000)
        layout.setRowMinimumHeight(0, 850)

        self.roiSizeEdit.textChanged.connect(self.imageWidget.updateROI)
        self.loadSTORMButton.clicked.connect(self.imageWidget.loadSTORM)
        self.loadSTEDButton.clicked.connect(self.imageWidget.loadSTED)

        self.filterImageButton.clicked.connect(self.imageWidget.imageFilter)
        self.dirButton.clicked.connect(self.imageWidget.getDirection)
        self.intThrButton.clicked.connect(self.imageWidget.intThreshold)
        self.corrButton.clicked.connect(self.imageWidget.corrMethodGUI)

    def keyPressEvent(self, event):
        key = event.key()

        if key == QtCore.Qt.Key_Left:
            self.imageWidget.roi.moveLeft()
        elif key == QtCore.Qt.Key_Right:
            self.imageWidget.roi.moveRight()
        elif key == QtCore.Qt.Key_Up:
            self.imageWidget.roi.moveUp()
        elif key == QtCore.Qt.Key_Down:
            self.imageWidget.roi.moveDown()


class ImageWidget(pg.GraphicsLayoutWidget):

    def __init__(self, main, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.main = main
        self.setWindowTitle('ImageGUI')
        self.subImgSize = 100

        # Item for displaying input image data
        self.inputVb = self.addViewBox(row=0, col=0)
        self.inputImg = pg.ImageItem()
        self.inputVb.addItem(self.inputImg)
        self.inputVb.setAspectLocked(True)

        # Contrast/color control
        self.inputImgHist = pg.HistogramLUTItem()
        self.inputImgHist.gradient.loadPreset('thermal')
        self.inputImgHist.setImageItem(self.inputImg)
        self.inputImgHist.vb.setLimits(yMin=0, yMax=20000)
        self.addItem(self.inputImgHist, row=0, col=1)

        # subimg
        self.subImg = pg.ImageItem()
        subimgHist = pg.HistogramLUTItem(image=self.subImg)
        subimgHist.gradient.loadPreset('thermal')
        self.addItem(subimgHist, row=1, col=1)

        self.subImgPlot = pg.PlotItem(title="Subimage")
        self.subImgPlot.addItem(self.subImg)
        self.addItem(self.subImgPlot, row=1, col=0)

        # Custom ROI for selecting an image region
        pxSize = np.float(self.main.STEDPxEdit.text())
        self.roi = tools.SubImgROI(self.main.subImgSize/pxSize)
        self.inputVb.addItem(self.roi)
        self.roi.setZValue(10)  # make sure ROI is drawn above image

        # Load sample STED image
        self.initialdir = os.getcwd()
        self.loadSTED(os.path.join(self.initialdir, 'labnanofisica',
                                   'ringfinder', 'spectrin1.tif'))

        # Correlation
        self.pCorr = pg.PlotItem(title="Correlation")
        self.pCorr.showGrid(x=True, y=True)
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

    def loadImage(self, pxSize, crop=0, filename=None):

        if not(isinstance(filename, str)):
            self.filename = utils.getFilename("Load image",
                                              [('Tiff file', '.tif')],
                                              self.initialdir)
        else:
            self.filename = filename

        if self.filename is not None:

            self.initialdir = os.path.split(self.filename)[0]
            self.pxSize = pxSize
            self.inputVb.clear()

            self.inputData = np.array(Image.open(self.filename))
            self.shape = self.inputData.shape
            self.inputData = self.inputData[crop:self.shape[0] - crop,
                                            crop:self.shape[1] - crop]
            self.shape = self.inputData.shape
            self.inputImg = pg.ImageItem()
            self.inputVb.addItem(self.inputImg)
            self.inputVb.setAspectLocked(True)
            self.inputImg.setImage(self.inputData)
            self.inputImgHist.setImageItem(self.inputImg)
            self.addItem(self.inputImgHist, row=0, col=1)
            self.inputVb.addItem(self.roi)

            # We need n 1um-sized subimages
            self.subimgPxSize = float(self.main.roiSizeEdit.text())/self.pxSize
            self.n = (np.array(self.shape)/self.subimgPxSize).astype(int)
            self.grid = tools.Grid(self.inputVb, self.shape, self.n)

            self.inputVb.setLimits(xMin=-0.05*self.shape[0],
                                   xMax=1.05*self.shape[0], minXRange=4,
                                   yMin=-0.05*self.shape[1],
                                   yMax=1.05*self.shape[1], minYRange=4)

            self.updateROI()
            self.updatePlot()

            self.dataMean = np.mean(self.inputData)
            self.dataStd = np.std(self.inputData)

            return True

        else:
            return False

    def loadSTED(self, filename=None):
        load = self.loadImage(np.float(self.main.STEDPxEdit.text()),
                              filename=filename)
        if load:
            self.main.sigmaEdit.setText('60')

    def loadSTORM(self, filename=None):
        # The STORM image has black borders because it's not possible to
        # localize molecules near the edge of the widefield image.
        # Therefore we need to crop those borders before running the analysis.
        nExcluded = np.float(self.main.excludedEdit.text())
        mag = np.float(self.main.magnificationEdit.text())
        load = self.loadImage(np.float(self.main.STORMPxEdit.text()),
                              crop=nExcluded*mag, filename=filename)
        if load:
            self.main.sigmaEdit.setText('200')
            self.inputImgHist.setLevels(0, 1)

    def updatePlot(self):
        self.selected = self.roi.getArrayRegion(self.inputData, self.inputImg)
        shape = self.selected.shape
        self.subImgSize = shape[0]
        self.subImgPlot.clear()
        self.subImg.setImage(self.selected)
        self.subImgPlot.addItem(self.subImg)
        self.subImgPlot.vb.setLimits(xMin=-0.05*shape[0], xMax=1.05*shape[0],
                                     yMin=-0.05*shape[1], yMax=1.05*shape[1],
                                     minXRange=4, minYRange=4)
        self.subImgPlot.vb.setRange(xRange=(0, shape[0]), yRange=(0, shape[1]))

    def updateROI(self):
        self.roiSize = np.float(self.main.roiSizeEdit.text()) / self.pxSize
        self.roi.setSize(self.roiSize, self.roiSize)
        self.roi.step = int(self.shape[0]/self.n[0])

    def corrMethodGUI(self):

        self.pCorr.clear()

        thr = np.float(self.main.intThrEdit.text())
        if np.any(self.selected > self.dataMean + thr*self.dataStd):

            self.getDirection()

            # for every subimg, we apply the correlation method for
            # ring finding
            corrThres = np.float(self.main.corrThresEdit.text())
            sigma = np.float(self.main.sigmaEdit.text()) / self.pxSize
            minLen = np.float(self.main.lineLengthEdit.text()) / self.pxSize
            thStep = np.float(self.main.thetaStepEdit.text())
            deltaTh = np.float(self.main.deltaThEdit.text())
            wvlen = np.float(self.main.wvlenEdit.text()) / self.pxSize
            sinPow = np.float(self.main.sinPowerEdit.text())
            args = [corrThres, sigma, minLen, thStep, deltaTh, wvlen, sinPow]
            output = tools.corrMethod(self.selected, *args, developer=True)
            self.th0, corrTheta, corrMax, thetaMax, phaseMax, rings = output

            if np.all([self.th0, corrMax]) is not None:
                self.bestAxon = simAxon(imSize=self.subImgSize, wvlen=wvlen,
                                        theta=thetaMax, phase=phaseMax,
                                        b=sinPow).data
                self.img1.setImage(self.bestAxon)
                self.img2.setImage(self.selected)

                # plot the threshold of correlation chosen by the user
                # phase steps are set to 20, TO DO: explore this parameter
                theta = np.arange(np.min([self.th0 - deltaTh, 0]), 180, thStep)
                pen1 = pg.mkPen(color=(0, 255, 100), width=2,
                                style=QtCore.Qt.SolidLine, antialias=True)
                pen2 = pg.mkPen(color=(255, 50, 60), width=1,
                                style=QtCore.Qt.SolidLine, antialias=True)
                self.pCorr.plot(theta, corrTheta, pen=pen1)
                self.pCorr.plot(theta, corrThres*np.ones(np.size(theta)),
                                pen=pen2)

                # plot the area within deltaTh from the found direction
                if self.th0 is not None:
                    thArea = np.arange(self.th0 - deltaTh, self.th0 + deltaTh)
                    if self.th0 < 0:
                        thArea += 180
                    self.pCorr.plot(thArea, 0.2*np.ones(len(thArea)),
                                    fillLevel=0, brush=(50, 50, 200, 100))

            if rings and np.abs(self.th0 - thetaMax) <= deltaTh:
                self.main.resultLabel.setText('<strong>MY PRECIOUS!<\strong>')
            else:
                if rings:
                    print('Correlation maximum outside direction theta range')
                self.main.resultLabel.setText('<strong>No rings<\strong>')
        else:
            print('Data below intensity threshold')
            self.main.resultLabel.setText('<strong>No rings<\strong>')

    def intThreshold(self):
        thr = np.float(self.main.intThrEdit.text())
        if np.any(self.selected > self.dataMean + thr*self.dataStd):
            print('NEURON')
        else:
            print('BACKGROUND')

    def imageFilter(self):
        ''' Removes background data from image.'''

        sigma = np.float(self.main.sigmaEdit.text())
        img = ndi.gaussian_filter(self.inputData, sigma)

        thresh = filters.threshold_otsu(img)
        binary = img > thresh

        self.inputImg.setImage(self.inputData * binary)

    def getDirection(self):

        sigma = np.float(self.main.sigmaEdit.text())
        minLen = np.float(self.main.lineLengthEdit.text())
        sigma /= self.pxSize
        minLen /= self.pxSize
        self.th0, lines = tools.getDirection(self.selected, sigma, minLen)

        print('Angle is {}, calculated from {} lines'.format(self.th0,
                                                             len(lines)))

        # Lines plot
        pen = pg.mkPen(color=(0, 255, 100), width=1, style=QtCore.Qt.SolidLine,
                       antialias=True)
        for line in lines:
            p0, p1 = line
            self.subImgPlot.plot((p0[1], p1[1]), (p0[0], p1[0]), pen=pen)

if __name__ == '__main__':
    app = QtGui.QApplication([])
    win = GollumDeveloper()
    win.show()
    app.exec_()
