# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 12:25:40 2016

@author: Luciano Masullo, Federico Barabas
"""

import os
import time
import numpy as np
from scipy import ndimage as ndi
import tifffile as tiff
import multiprocessing as mp

from PIL import Image
import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

import labnanofisica.utils as utils
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
        batchSTORMAction = QtGui.QAction('Analyse batch of STORM images...',
                                         self)
        batchSTEDAction = QtGui.QAction('Analyse batch of STED images...',
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

        outputImgHist = pg.HistogramLUTItem()
        outputImgHist.gradient.loadPreset('thermal')
        outputImgHist.setImageItem(self.outputImg)
        outputImgHist.vb.setLimits(yMin=0, yMax=20000)
        self.outputWidget.addItem(outputImgHist)

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
        loadFrame.setFixedHeight(200)

        # Ring finding method settings frame
        self.intThrLabel = QtGui.QLabel('#sigmas threshold from mean')
        self.intThresEdit = QtGui.QLineEdit('0.2')
        gaussianSigmaLabel = QtGui.QLabel('Gaussian filter sigma [nm]')
        self.sigmaEdit = QtGui.QLineEdit('60')
        minLenLabel = QtGui.QLabel('Direction lines min length [nm]')
        self.lineLengthEdit = QtGui.QLineEdit('300')
        self.corrThresEdit = QtGui.QLineEdit('0.075')
        self.thetaStepEdit = QtGui.QLineEdit('3')
        self.deltaAngleEdit = QtGui.QLineEdit('20')
        powLabel = QtGui.QLabel('Sinusoidal pattern power')
        self.sinPowerEdit = QtGui.QLineEdit('3')
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
        settingsFrame.setFixedHeight(280)

        buttonsLayout = QtGui.QGridLayout()
        self.buttonWidget.setLayout(buttonsLayout)
        buttonsLayout.addWidget(loadFrame, 0, 0)
        buttonsLayout.addWidget(settingsFrame, 1, 0)

        self.loadSTORMButton.clicked.connect(self.loadSTORM)
        self.loadSTEDButton.clicked.connect(self.loadSTED)

        self.corrButton.clicked.connect(self.ringFinder)

        # Load sample STED image
        self.initialdir = os.getcwd()
        self.loadSTED(os.path.join(self.initialdir, 'labnanofisica',
                                   'ringfinder', 'spectrinSTED.tif'))

    def loadSTED(self, filename=None):
        load = self.loadImage(np.float(self.STEDPxEdit.text()),
                              filename=filename)
        if load:
            self.sigmaEdit.setText('60')

    def loadSTORM(self, filename=None):
        # The STORM image has black borders because it's not possible to
        # localize molecules near the edge of the widefield image.
        # Therefore we need to crop those borders before running the analysis.
        nExcluded = np.float(self.excludedEdit.text())
        mag = np.float(self.magnificationEdit.text())
        load = self.loadImage(np.float(self.STORMPxEdit.text()),
                              crop=nExcluded*mag, filename=filename)
        if load:
            self.sigmaEdit.setText('200')
            self.inputImgHist.setLevels(0, 0.3)

    def loadImage(self, pxSize, crop=0, filename=None):

        try:
            if not(isinstance(filename, str)):
                self.filename = utils.getFilename("Load image",
                                                  [('Tiff file', '.tif')],
                                                  self.initialdir)
            else:
                self.filename = filename

            if self.filename is not None:

                self.initialdir = os.path.split(self.filename)[0]
                self.crop = crop
                self.pxSize = pxSize
                self.inputVb.clear()

                im = Image.open(self.filename)
                self.inputData = np.array(im).astype(np.float64)
                self.initShape = self.inputData.shape
                self.inputData = self.inputData[crop:self.initShape[0] - crop,
                                                crop:self.initShape[1] - crop]
                self.shape = self.inputData.shape
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

                return True

            else:
                return False

        except OSError:
            print("No file selected!")

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
        blocksInput = tools.blockshaped(self.inputData,
                                        self.inputData.shape[0]/m[0],
                                        self.inputData.shape[1]/m[1])

        # for every subimg, we apply the correlation method for ring finding
        intThres = np.float(self.intThresEdit.text())
        corrThres = np.float(self.corrThresEdit.text())
        sigma = np.float(self.sigmaEdit.text())/self.pxSize
        minLen = np.float(self.lineLengthEdit.text())/self.pxSize
        thetaStep = np.float(self.deltaAngleEdit.text())
        deltaTh = np.float(self.deltaAngleEdit.text())
        wvlen = np.float(self.wvlenEdit.text())/self.pxSize
        sinPow = np.float(self.sinPowerEdit.text())

        # We apply intensity threshold to smoothed data so we don't catch
        # tiny bright spots outside neurons
        inputDataS = ndi.gaussian_filter(self.inputData, 300/self.pxSize)
        meanS = np.mean(inputDataS)
        stdS = np.std(inputDataS)
        blocksInputS = tools.blockshaped(inputDataS, inputDataS.shape[0]/m[0],
                                         inputDataS.shape[1]/m[1])
        # Multi-core code
        cpus = mp.cpu_count()
        step = len(blocksInput) // cpus
        chunks = [[i*step, (i + 1)*step] for i in np.arange(cpus)]
        chunks[-1][1] = len(blocksInput)
        # Correlation arguments
        cArgs = corrThres, sigma, minLen, thetaStep, deltaTh, wvlen, sinPow
        # Finder arguments
        fArg = meanS, stdS, intThres, cArgs
        args = [[blocksInput[i:j], blocksInputS[i:j], fArg] for i, j in chunks]
        pool = mp.Pool(processes=cpus)
        results = pool.map(chunkFinder, args)
        pool.close()
        pool.join()
        self.localCorr = np.concatenate(results[:])
        M = self.localCorr > corrThres

        # code for visualization of the output
        M1 = M.reshape(*m)
        self.outputData = np.kron(M1, np.ones((self.inputData.shape[0]/m[0],
                                               self.inputData.shape[0]/m[1])))
        showIm = np.fliplr(np.transpose(self.inputData))
        self.outputImg.setImage(showIm)
        showIm = np.fliplr(np.transpose(self.outputData))
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

    def batch(self, function):
        try:
            filenames = utils.getFilenames("Load images",
                                           [('Tiff file', '.tif')],
                                           self.initialdir)
            nfiles = len(filenames)
            function(filenames[0])
            corrArray = np.zeros((nfiles, self.n[0], self.n[1]))
            print('Processing folder ' + os.path.split(filenames[0])[0])
            t0 = time.time()
            for i in np.arange(len(filenames)):
                print(os.path.split(filenames[i])[1])
                function(filenames[i])
                self.ringFinder(False)
                corr = self.localCorr.reshape(*self.n)
                corrArray[i] = corr
                corrName = utils.insertSuffix(filenames[i], '_correlation')

                # Expand correlation array so it matches data shape
                m = self.shape/self.n
                corr = np.repeat(np.repeat(corr, m[1], axis=1), m[0], axis=0)
                im = np.empty(self.initShape, dtype=np.single)
                im[:] = np.NAN
                im[self.crop:self.initShape[0] - self.crop,
                   self.crop:self.initShape[1] - self.crop] = corr

                tiff.imsave(corrName, im, software='Gollum', imagej=True,
                            resolution=(1000/self.pxSize, 1000/self.pxSize),
                            metadata={'spacing': 1, 'unit': 'um'})

            print('Done in {.0} seconds'.format(time.time() - t0))

            # plot histogram of the correlation values
            plotData = np.nan_to_num(corrArray.flatten())
            plt.figure(0)
            plt.hist(plotData, bins=30, range=(0.0001, np.max(plotData)))
            plt.title("Correlations Histogram")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.show()

        except IndexError:
            print("No file selected!")

    def batchSTORM(self):
        self.batch(self.loadSTORM)

    def batchSTED(self):
        self.batch(self.loadSTED)


def chunkFinder(args):

    blocks, blocksS, fArgs = args
    meanS, stdS, intThres, cArgs = fArgs
#    cArgs = corrThres, sigma, minLen, thetaStep, deltaTh, wvlen, sinPow

    localCorr = np.zeros(len(blocks))

    for i in np.arange(len(blocks)):
        block = blocks[i]
        blockS = blocksS[i]
        rings = False

        if np.any(blockS > meanS + intThres*stdS):
            output = tools.corrMethod(block, *cArgs)
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
