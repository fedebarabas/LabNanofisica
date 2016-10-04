# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 12:25:40 2016

@author: Luciano Masullo, Federico Barabas
"""

import os
import time
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import curve_fit
from scipy.stats import norm
try:
    import skimage.filters as filters
except ImportError:
    import skimage.filter as filters
import tifffile as tiff
import multiprocessing as mp

from PIL import Image
import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

import labnanofisica.utils as utils
import labnanofisica.gaussians as gaussians
import labnanofisica.ringfinder.tools as tools


class Gollum(QtGui.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.i = 0

        self.setWindowTitle('Gollum: the Ring Finder')

        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&Run')
        batchSTORMAction = QtGui.QAction('Analyze batch of STORM images...',
                                         self)
        batchSTEDAction = QtGui.QAction('Analyze batch of STED images...',
                                        self)

        batchSTORMAction.triggered.connect(self.batchSTORM)
        batchSTEDAction.triggered.connect(self.batchSTED)
        fileMenu.addAction(batchSTORMAction)
        fileMenu.addAction(batchSTEDAction)
        fileMenu.addSeparator()

        exitAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(QtGui.QApplication.closeAllWindows)
        fileMenu.addAction(exitAction)

        # Main Widgets' layout
        self.mainLayout = QtGui.QGridLayout()
        self.cwidget.setLayout(self.mainLayout)

        # input, output and buttons widgets
        self.inputWidget = pg.GraphicsLayoutWidget()
        self.outputWidget = pg.GraphicsLayoutWidget()
        self.buttonWidget = QtGui.QWidget()

        # layout of the three widgets
        self.mainLayout.addWidget(self.inputWidget, 0, 1)
        self.mainLayout.addWidget(self.outputWidget, 0, 2)
        self.mainLayout.addWidget(self.buttonWidget, 0, 0)
        self.mainLayout.setColumnMinimumWidth(1, 600)
        self.mainLayout.setColumnMinimumWidth(2, 600)

        self.inputImgItem = pg.ImageItem()
        self.inputVb = self.inputWidget.addViewBox(col=0, row=0)
        self.inputVb.setAspectLocked(True)
        self.inputVb.addItem(self.inputImgItem)

        self.inputImgHist = pg.HistogramLUTItem()
        self.inputImgHist.gradient.loadPreset('thermal')
        self.inputImgHist.setImageItem(self.inputImgItem)
        self.inputImgHist.vb.setLimits(yMin=0, yMax=20000)
        self.inputWidget.addItem(self.inputImgHist)

        self.outputImg = pg.ImageItem()
        self.outputVb = self.outputWidget.addViewBox(col=0, row=0)
        self.outputVb.setAspectLocked(True)
        self.outputVb.addItem(self.outputImg)

        self.outputImgHist = pg.HistogramLUTItem()
        self.outputImgHist.gradient.loadPreset('thermal')
        self.outputImgHist.setImageItem(self.outputImg)
        self.outputImgHist.vb.setLimits(yMin=0, yMax=20000)
        self.outputWidget.addItem(self.outputImgHist)

        self.outputResult = pg.ImageItem()
        self.outputVb.addItem(self.outputResult)

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
        loadLayout.setColumnMinimumWidth(1, 40)
        loadFrame.setFixedHeight(200)

        # Ring finding method settings frame
        self.intThrLabel = QtGui.QLabel('#sigmas threshold from mean')
        self.intThresEdit = QtGui.QLineEdit('0.5')
        gaussianSigmaLabel = QtGui.QLabel('Gaussian filter sigma [nm]')
        self.sigmaEdit = QtGui.QLineEdit('150')
        minLenLabel = QtGui.QLabel('Direction lines min length [nm]')
        self.lineLengthEdit = QtGui.QLineEdit('300')
        self.corrThresEdit = QtGui.QLineEdit('0.075')
        self.thetaStepEdit = QtGui.QLineEdit('3')
        self.deltaAngleEdit = QtGui.QLineEdit('20')
        powLabel = QtGui.QLabel('Sinusoidal pattern power')
        self.sinPowerEdit = QtGui.QLineEdit('6')
        wvlenLabel = QtGui.QLabel('wvlen of corr pattern [nm]')
        self.wvlenEdit = QtGui.QLineEdit('180')
        self.corrButton = QtGui.QPushButton('Correlation')
        settingsFrame = QtGui.QFrame(self)
        settingsFrame.setFrameStyle(QtGui.QFrame.Panel)
        settingsLayout = QtGui.QGridLayout()
        settingsFrame.setLayout(settingsLayout)
        settingsTitle = QtGui.QLabel('<strong>Ring finding settings</strong>')
        settingsTitle.setTextFormat(QtCore.Qt.RichText)
        settingsLayout.addWidget(settingsTitle, 0, 0)
        settingsLayout.addWidget(self.intThrLabel, 1, 0)
        settingsLayout.addWidget(self.intThresEdit, 1, 1)
        settingsLayout.addWidget(gaussianSigmaLabel, 2, 0)
        settingsLayout.addWidget(self.sigmaEdit, 2, 1)
        settingsLayout.addWidget(minLenLabel, 3, 0)
        settingsLayout.addWidget(self.lineLengthEdit, 3, 1)
        settingsLayout.addWidget(QtGui.QLabel('Correlation threshold'), 4, 0)
        settingsLayout.addWidget(self.corrThresEdit, 4, 1)
        settingsLayout.addWidget(QtGui.QLabel('Angular step [°]'), 5, 0)
        settingsLayout.addWidget(self.thetaStepEdit, 5, 1)
        settingsLayout.addWidget(QtGui.QLabel('Delta Angle [°]'), 6, 0)
        settingsLayout.addWidget(self.deltaAngleEdit, 6, 1)
        settingsLayout.addWidget(powLabel, 7, 0)
        settingsLayout.addWidget(self.sinPowerEdit, 7, 1)
        settingsLayout.addWidget(wvlenLabel, 8, 0)
        settingsLayout.addWidget(self.wvlenEdit, 8, 1)
        settingsLayout.addWidget(self.corrButton, 9, 0, 1, 2)
        loadLayout.setColumnMinimumWidth(1, 40)
        settingsFrame.setFixedHeight(280)

        buttonsLayout = QtGui.QGridLayout()
        self.buttonWidget.setLayout(buttonsLayout)
        buttonsLayout.addWidget(loadFrame, 0, 0)
        buttonsLayout.addWidget(settingsFrame, 1, 0)

        self.loadSTORMButton.clicked.connect(self.loadSTORM)
        self.loadSTEDButton.clicked.connect(self.loadSTED)
        self.sigmaEdit.textChanged.connect(self.updateImage)
        self.corrButton.clicked.connect(self.ringFinder)

        # Load sample STED image
        self.initialdir = os.getcwd()
        self.loadSTED(os.path.join(self.initialdir, 'labnanofisica',
                                   'ringfinder', 'spectrinSTED.tif'))

    def loadSTED(self, filename=None):
        load = self.loadImage(np.float(self.STEDPxEdit.text()), 'STED',
                              filename=filename)
        if load:
            self.sigmaEdit.setText('100')
            self.intThresEdit.setText('0.1')

    def loadSTORM(self, filename=None):
        # The STORM image has black borders because it's not possible to
        # localize molecules near the edge of the widefield image.
        # Therefore we need to crop those borders before running the analysis.
        nExcluded = np.float(self.excludedEdit.text())
        mag = np.float(self.magnificationEdit.text())
        load = self.loadImage(np.float(self.STORMPxEdit.text()), 'STORM',
                              crop=nExcluded*mag, filename=filename)
        if load:
            self.inputImgHist.setLevels(0, 0.5)
            self.sigmaEdit.setText('150')
            self.intThresEdit.setText('0.5')

    def loadImage(self, pxSize, tt, crop=0, filename=None):

        try:

            if not(isinstance(filename, str)):
                self.filename = utils.getFilename('Load ' + tt + ' image',
                                                  [('Tiff file', '.tif')],
                                                  self.initialdir)
            else:
                self.filename = filename

            if self.filename is not None:

                self.initialdir = os.path.split(self.filename)[0]
                self.crop = np.int(crop)
                self.pxSize = pxSize
                self.inputVb.clear()
                self.outputVb.clear()
                self.outputImg.clear()
                self.outputResult.clear()

                im = Image.open(self.filename)
                self.inputData = np.array(im).astype(np.float64)
                self.initShape = self.inputData.shape
                bound = (np.array(self.initShape) - self.crop).astype(np.int)
                self.inputData = self.inputData[self.crop:bound[0],
                                                self.crop:bound[1]]
                self.shape = self.inputData.shape
                self.updateImage()
                self.inputVb.addItem(self.inputImgItem)
                showIm = np.fliplr(np.transpose(self.inputData))
                self.inputImgItem.setImage(showIm)

                # We need 1um n-sized subimages
                self.subimgPxSize = 1000/self.pxSize
                self.n = (np.array(self.shape)/self.subimgPxSize).astype(int)
                self.grid = tools.Grid(self.inputVb, self.shape, self.n)

                self.inputVb.setLimits(xMin=-0.05*self.shape[0],
                                       xMax=1.05*self.shape[0], minXRange=4,
                                       yMin=-0.05*self.shape[1],
                                       yMax=1.05*self.shape[1], minYRange=4)

                self.dataMean = np.mean(self.inputData)
                self.dataStd = np.std(self.inputData)

                self.outputVb.addItem(self.outputImg)
                self.outputWidget.addItem(self.outputImgHist)
                self.outputVb.addItem(self.outputResult)

                return True

            else:
                return False

        except OSError:
            print("No file selected!")

    def updateImage(self):
        self.gaussSigma = np.float(self.sigmaEdit.text())/self.pxSize
        self.inputDataS = ndi.gaussian_filter(self.inputData,
                                              self.gaussSigma)
        self.meanS = np.mean(self.inputDataS)
        self.stdS = np.std(self.inputDataS)

        self.showImS = np.fliplr(np.transpose(self.inputDataS))

        # binarization of image
#        thresh = filters.threshold_otsu(self.inputDataS)
#        self.mask = self.inputDataS < thresh
        thr = np.float(self.intThresEdit.text())
        self.mask = self.inputDataS < self.meanS + thr*self.stdS
        self.showMask = np.fliplr(np.transpose(self.mask))

    def ringFinder(self, show=True):
        """RingFinder handles the input data, and then evaluates every subimg
        using the given algorithm which decides if there are rings or not.
        Subsequently gives the output data and plots it"""

        # initialize variables
        self.outputImg.clear()
        self.outputResult.clear()

        # m is such that the image has m x m subimages
        m = self.n

        # shape the data into the subimg that we need for the analysis
        nblocks = np.array(self.inputData.shape)/m
        blocksInput = tools.blockshaped(self.inputData, *nblocks)
        blocksInputS = tools.blockshaped(self.inputDataS, *nblocks)
        blocksMask = tools.blockshaped(self.mask, *nblocks)

        # for every subimg, we apply the correlation method for ring finding
        intThres = np.float(self.intThresEdit.text())
        corrThres = np.float(self.corrThresEdit.text())
        minLen = np.float(self.lineLengthEdit.text())/self.pxSize
        thetaStep = np.float(self.deltaAngleEdit.text())
        deltaTh = np.float(self.deltaAngleEdit.text())
        wvlen = np.float(self.wvlenEdit.text())/self.pxSize
        sinPow = np.float(self.sinPowerEdit.text())

        # Multi-core code
        cpus = mp.cpu_count()
        step = len(blocksInput) // cpus
        chunks = [[i*step, (i + 1)*step] for i in np.arange(cpus)]
        chunks[-1][1] = len(blocksInput)
        # Correlation arguments
        cArgs = corrThres, minLen, thetaStep, deltaTh, wvlen, sinPow
        # Finder arguments
        fArg = self.meanS, self.stdS, intThres, cArgs
        args = [[blocksInput[i:j], blocksInputS[i:j], blocksMask[i:j], fArg]
                for i, j in chunks]
        pool = mp.Pool(processes=cpus)
        results = pool.map(chunkFinder, args)
        pool.close()
        pool.join()
        self.localCorr = np.nan_to_num(np.concatenate(results[:]))
#        self.localCorr -= np.min(self.localCorr)
        self.localCorr = self.localCorr.reshape(*self.n)

        # code for visualization of the output
        mag = np.array(self.inputData.shape)/self.n
        self.localCorrBig = np.repeat(self.localCorr, mag[0], 0)
        self.localCorrBig = np.repeat(self.localCorrBig, mag[1], 1)
        self.outputImg.setImage(np.fliplr(np.transpose(self.inputData)))
        showIm = 100*np.fliplr(np.transpose(self.localCorrBig))
        self.outputResult.setImage(showIm)
        self.outputResult.setZValue(10)     # make sure this image is on top
        self.outputResult.setOpacity(0.5)

        self.outputVb.setLimits(xMin=-0.05*self.shape[0],
                                xMax=1.05*self.shape[0], minXRange=4,
                                yMin=-0.05*self.shape[1],
                                yMax=1.05*self.shape[1], minYRange=4)
        tools.Grid(self.outputVb, self.shape, self.n)

        if show:
            plt.figure()
            data = self.localCorr.reshape(*m)
            data = np.rot90(data)
            data = np.flipud(data)
            heatmap = plt.pcolor(data)

            for y in range(data.shape[0]):
                for x in range(data.shape[1]):
                    plt.text(x + 0.5, y + 0.5, '%.4f' % data[y, x],
                             horizontalalignment='center',
                             verticalalignment='center',)

            plt.colorbar(heatmap)

            plt.show()

    def batch(self, function, tech):
        try:
            filenames = utils.getFilenames('Load ' + tech + ' images',
                                           [('Tiff file', '.tif')],
                                           self.initialdir)
            nfiles = len(filenames)
            function(filenames[0])
            corrArray = np.zeros((nfiles, self.n[0], self.n[1]))

            # Expand correlation array so it matches data shape
            m = self.shape/self.n
            corrExp = np.empty((nfiles, self.initShape[0], self.initShape[1]),
                               dtype=np.single)
            corrExp[:] = np.NAN

            path = os.path.split(filenames[0])[0]
            folder = os.path.split(path)[1]
            print('Processing folder', path)
            t0 = time.time()
            for i in np.arange(nfiles):
                print(os.path.split(filenames[i])[1])
                function(filenames[i])
                self.ringFinder(False)
                corrArray[i] = self.localCorr

                bound = (np.array(self.initShape) - self.crop).astype(np.int)
                corrExp[i, self.crop:bound[0],
                        self.crop:bound[1]] = self.localCorrBig

                # Save correlation values array
                corrName = utils.insertSuffix(filenames[i], '_correlation')
                tiff.imsave(corrName, corrExp[i], software='Gollum',
                            imagej=True,
                            resolution=(1000/self.pxSize, 1000/self.pxSize),
                            metadata={'spacing': 1, 'unit': 'um'})

            print('Done in {0:.0f} seconds'.format(time.time() - t0))

            # plot histogram of the correlation values
            corrArray = np.nan_to_num(corrArray)
            hRange = (0.0001, np.max(corrArray))
            y, x, _ = plt.hist(corrArray.flatten(), bins=60, range=hRange)
            x = (x[1:] + x[:-1])/2
            plt.title("Correlations Histogram")
            plt.xlabel("Value")
            plt.ylabel("Frequency")

            # Bimodal fitting
            expected = (0.10, 0.05, np.max(y[:len(x)//2]),
                        0.20, 0.05, np.max(y[len(x)//2:]))
            params, cov = curve_fit(gaussians.bimodal, x, y, expected)
            threshold = norm.ppf(0.90, *params[:2])
            ringsRatio = np.sum(y[x > threshold]) / np.sum(y)
            print('Rings threshold:', np.round(threshold, 2))
#            threshold = 0.1

            # Save boolean images (rings or no rings)
            ringsArray = corrArray > threshold
            ringsExp = np.zeros((nfiles, self.initShape[0], self.initShape[1]),
                                dtype=bool)
            for i in np.arange(len(filenames)):
                expanded = np.repeat(np.repeat(ringsArray[i], m[1], axis=1),
                                     m[0], axis=0)
                ringsExp[i, self.crop:self.initShape[0] - self.crop,
                         self.crop:self.initShape[1] - self.crop] = expanded
                tiff.imsave(utils.insertSuffix(filenames[i], '_rings'),
                            ringsExp[i].astype(np.single), software='Gollum',
                            imagej=True,
                            resolution=(1000/self.pxSize, 1000/self.pxSize),
                            metadata={'spacing': 1, 'unit': 'um'})

            # Plotting
            plt.figure(0)
            plt.bar(x, y, align='center', width=(x[1] - x[0]))
            plt.plot(x, gaussians.bimodal(x, *params), color='red', lw=3,
                     label='model')
            plt.plot(x, gaussians.gauss(x, *params[:3]), color='green', lw=3,
                     label='no rings')
            plt.plot(x, gaussians.gauss(x, *params[-3:]), color='black', lw=3,
                     label='rings')
            plt.legend()

            plt.savefig(os.path.join(path, folder + 'corr_hist'))
            plt.close()

#            np.save(os.path.join(path, 'histx'), x)
#            np.save(os.path.join(path, 'histy'), y)

        except IndexError:
            print("No file selected!")

    def batchSTORM(self):
        self.batch(self.loadSTORM, 'STORM')

    def batchSTED(self):
        self.batch(self.loadSTED, 'STED')


def chunkFinder(args):

    blocks, blocksS, blocksMask, fArgs = args
    meanS, stdS, intThres, cArgs = fArgs
#    cArgs = corrThres, sigma, minLen, thetaStep, deltaTh, wvlen, sinPow

    localCorr = np.zeros(len(blocks))

    for i in np.arange(len(blocks)):
        block = blocks[i]
        blockS = blocksS[i]
        mask = blocksMask[i]
        rings = False

        # Block may be excluded from the analysis for two reasons. Firstly,
        # because the intensity for all its pixels may be too low. Secondly,
        # because the part of the block that belongs to a neuron may be below
        # an arbitrary 30% of the block.
        # We apply intensity threshold to smoothed data so we don't catch
        # tiny bright spots outside neurons
        neuronFrac = 1 - np.sum(mask)/np.size(mask)
        if np.any(blockS > meanS + intThres*stdS) and neuronFrac > 0.25:
            output = tools.corrMethod(block, mask, *cArgs)
            angle, corrTheta, corrMax, theta, phase, rings = output

            # Store results
            localCorr[i] = corrMax
        else:
            localCorr[i] = None

    return localCorr


if __name__ == '__main__':
    app = QtGui.QApplication([])
    win = Gollum()
    win.show()
    app.exec_()
