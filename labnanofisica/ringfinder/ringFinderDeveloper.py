# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 13:41:48 2014

@authors: Luciano Masullo, Federico Barabas
"""

import os
import numpy as np

from scipy import ndimage as ndi
import skimage.filters as filters
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

#        self.setFixedSize(1275, 800)

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
        self.lineLengthEdit = QtGui.QLineEdit('300')
        self.wvlenEdit = QtGui.QLineEdit('180')

        self.roiLabel = QtGui.QLabel('ROI size (px)')
        self.corrThresLabel = QtGui.QLabel('Correlation threshold')
        self.thetaStepLabel = QtGui.QLabel('Angular step (°)')
        self.deltaThLabel = QtGui.QLabel('Delta Angle (°)')
        self.sigmaLabel = QtGui.QLabel('Sigma of gaussian filter (nm)')
        self.sinPowerLabel = QtGui.QLabel('Sinusoidal pattern power')
        self.pxSizeLabel = QtGui.QLabel('Pixel size (nm)')
        self.intThrLabel = QtGui.QLabel('# of times from mean intensity')
        self.lineLengthLabel = QtGui.QLabel('Direction lines min length [nm]')
        # Direction lines lengths are expressed in fraction of subimg size
        self.wvlenLabel = QtGui.QLabel('wvlen of corr pattern (nm)')

        settingsFrame = QtGui.QFrame(self)
        settingsFrame.setFrameStyle(QtGui.QFrame.Panel)
        settingsLayout = QtGui.QGridLayout()
        settingsFrame.setLayout(settingsLayout)
        settingsTitle = QtGui.QLabel('<strong>Ring finding settings</strong>')
        settingsTitle.setTextFormat(QtCore.Qt.RichText)
        settingsLayout.addWidget(settingsTitle, 0, 0)
        settingsLayout.addWidget(self.roiLabel, 1, 0)
        settingsLayout.addWidget(self.roiSizeEdit, 1, 1)
        settingsLayout.addWidget(self.corrButton, 2, 0, 1, 2)
        settingsLayout.addWidget(self.corrThresLabel, 3, 0)
        settingsLayout.addWidget(self.corrThresEdit, 3, 1)
        settingsLayout.addWidget(self.wvlenLabel, 4, 0)
        settingsLayout.addWidget(self.wvlenEdit, 4, 1)
        settingsLayout.addWidget(self.sinPowerLabel, 5, 0)
        settingsLayout.addWidget(self.sinPowerEdit, 5, 1)
        settingsLayout.addWidget(self.thetaStepLabel, 6, 0)
        settingsLayout.addWidget(self.thetaStepEdit, 6, 1)
        settingsLayout.addWidget(self.deltaThLabel, 7, 0)
        settingsLayout.addWidget(self.deltaThEdit, 7, 1)
        settingsLayout.addWidget(self.dirButton, 8, 0, 1, 2)
        settingsLayout.addWidget(self.lineLengthLabel, 9, 0)
        settingsLayout.addWidget(self.lineLengthEdit, 9, 1)
        settingsLayout.addWidget(self.sigmaLabel, 10, 0)
        settingsLayout.addWidget(self.sigmaEdit, 10, 1)
        settingsLayout.addWidget(self.filterImageButton, 11, 0, 1, 2)
        settingsLayout.addWidget(self.intThrButton, 12, 0, 1, 2)
        settingsLayout.addWidget(self.intThrLabel, 13, 0)
        settingsLayout.addWidget(self.intThrEdit, 13, 1)
        settingsLayout.addWidget(self.stormCropButton, 14, 0, 1, 2)
        settingsFrame.setFixedHeight(420)

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

        self.roiSizeEdit.textChanged.connect(self.imageWidget.updateROI)
        self.loadSTORMButton.clicked.connect(self.imageWidget.loadSTORM)
        self.loadSTEDButton.clicked.connect(self.imageWidget.loadSTED)

        self.filterImageButton.clicked.connect(self.imageWidget.imageFilter)
        self.dirButton.clicked.connect(self.imageWidget.getDirection)
        self.intThrButton.clicked.connect(self.imageWidget.intThreshold)
        self.stormCropButton.clicked.connect(self.imageWidget.stormCrop)


class ImageWidget(pg.GraphicsLayoutWidget):

    def __init__(self, main, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.main = main
        self.setWindowTitle('ImageGUI')
        self.subimgSize = 100

        # Item for displaying 10x10 image data
        self.inputVb = self.addViewBox(row=0, col=0)
        self.bigimg = pg.ImageItem()
        self.inputVb.addItem(self.bigimg)
        self.inputVb.setAspectLocked(True)

        # Custom ROI for selecting an image region
        self.roi = pg.ROI([0, 0], [0, 0])
        self.roiSizeX = np.float(self.main.roiSizeEdit.text())
        self.roiSizeY = np.float(self.main.roiSizeEdit.text())
        self.roi.setSize(self.roiSizeX, self.roiSizeY)
        self.roi.addScaleHandle([1, 1], [0, 0], lockAspect=True)
        self.inputVb.addItem(self.roi)
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
        self.inputData = np.array(self.im)
        self.bigimg.setImage(self.inputData)
        self.bigimg_hist.setLevels(self.inputData.min(), self.inputData.max())

        self.grid = tools.Grid(self.inputVb, self.inputData)

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

        self.inputVb.clear()
        pxSize = np.float(self.main.pxSizeEdit.text())
        subimgPxSize = 1000/pxSize
        self.filename = utils.getFilename("Load image",
                                          [('Tiff file', '.tif')])
        self.inputData = np.array(Image.open(self.filename))
        self.bigimg = pg.ImageItem()
        self.inputVb.addItem(self.bigimg)
        self.inputVb.setAspectLocked(True)
        self.bigimg.setImage(self.inputData)
        self.bigimg_hist.setImageItem(self.bigimg)
        self.addItem(self.bigimg_hist, row=0, col=1)
        self.inputVb.addItem(self.roi)
        self.setGrid(np.int(np.shape(self.inputData)[0]/subimgPxSize))

        self.inputVb.setLimits(xMin=-0.05*self.inputData.shape[0],
                               xMax=1.05*self.inputData.shape[0], minXRange=4,
                               yMin=-0.05*self.inputData.shape[1],
                               yMax=1.05*self.inputData.shape[1], minYRange=4)

    def loadSTED(self):
        self.loadImage(np.float(self.STEDPxEdit.text()))
        self.pxSize = np.float(self.STEDMPxEdit.text())

    def loadSTORM(self):
        # The STORM image has black borders because it's not possible to
        # localize molecules near the edge of the widefield image.
        # Therefore we need to crop those borders before running the analysis.
        nExcluded = np.float(self.excludedEdit.text())
        mag = np.float(self.magnificationEdit.text())
        self.loadImage(np.float(self.STORMPxEdit.text()), crop=nExcluded*mag)
        self.pxSize = np.float(self.STORMPxEdit.text())

    def updatePlot(self):

        self.dirPlot.clear()
        self.selected = self.roi.getArrayRegion(self.inputData, self.bigimg)
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
        intMean = np.mean(self.inputData)
        if np.mean(self.selected) < intMean/thr:
            print('BACKGROUND')
        else:
            print('NEURON')

    def imageFilter(self):

        sigma = np.float(self.main.sigmaEdit.text())
        img = ndi.gaussian_filter(self.inputData, sigma)

        thresh = filters.threshold_otsu(img)
        binary = img > thresh

        self.inputData *= binary
        self.bigimg.setImage(self.inputData)

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

        self.inputData = self.inputData[59:-59, 59:-59]
        self.inputVb.clear()
        self.inputVb.addItem(self.bigimg)
        self.inputVb.setAspectLocked(True)
        self.bigimg.setImage(self.inputData)
        self.bigimg_hist.setImageItem(self.bigimg)
        self.addItem(self.bigimg_hist, row=0, col=1)
        self.inputVb.addItem(self.roi)
        pxSize = np.float(self.main.pxSizeEdit.text())
        subimgPxSize = 1000/pxSize
        tools.setGrid(self.inputVb, self.inputData,
                      np.int(np.shape(self.inputData)[0]/subimgPxSize))

if __name__ == '__main__':

    app = QtGui.QApplication([])

    win = GollumDeveloper()
    win.show()
    app.exec_()
