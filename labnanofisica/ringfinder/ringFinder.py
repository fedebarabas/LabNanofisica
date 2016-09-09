# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 12:25:40 2016

@author: Luciano Masullo, Federico Barabas
"""

import os
import numpy as np

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

        # number of subimg (side)
        self.subimgNum = 10

        self.setWindowTitle('Gollum: the Ring Finder')

        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)

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

        inputImgHist = pg.HistogramLUTItem()
        inputImgHist.gradient.loadPreset('thermal')
        inputImgHist.setImageItem(self.inputImgItem)
        inputImgHist.vb.setLimits(yMin=0, yMax=20000)
        self.inputWidget.addItem(inputImgHist)

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
        self.corrButton = QtGui.QPushButton('Correlation')
        self.deltaAngleEdit = QtGui.QLineEdit('30')
        self.thetaStepEdit = QtGui.QLineEdit('3')
        self.fftThrEdit = QtGui.QLineEdit('0.6')
        gaussianSigmaLabel = QtGui.QLabel('Gaussian filter sigma [nm]')
        self.sigmaEdit = QtGui.QLineEdit('60')
        minLenLabel = QtGui.QLabel('Direction lines min length [nm]')
        self.lineLengthEdit = QtGui.QLineEdit('300')
        self.corr2thrEdit = QtGui.QLineEdit('0.1')
        powLabel = QtGui.QLabel('Sinusoidal pattern power')
        self.sinPowerEdit = QtGui.QLineEdit('2')
        wvlenLabel = QtGui.QLabel('wvlen of corr pattern [nm]')
        self.wvlenEdit = QtGui.QLineEdit('180')
        settingsFrame = QtGui.QFrame(self)
        settingsFrame.setFrameStyle(QtGui.QFrame.Panel)
        settingsLayout = QtGui.QGridLayout()
        settingsFrame.setLayout(settingsLayout)
        settingsTitle = QtGui.QLabel('<strong>Ring finding settings</strong>')
        settingsTitle.setTextFormat(QtCore.Qt.RichText)
        settingsLayout.addWidget(settingsTitle, 0, 0)
        settingsLayout.addWidget(gaussianSigmaLabel, 1, 0)
        settingsLayout.addWidget(self.sigmaEdit, 1, 1)
        settingsLayout.addWidget(minLenLabel, 2, 0)
        settingsLayout.addWidget(self.lineLengthEdit, 2, 1)
        settingsLayout.addWidget(QtGui.QLabel('Correlation threshold'), 3, 0)
        settingsLayout.addWidget(self.corr2thrEdit, 3, 1)
        settingsLayout.addWidget(QtGui.QLabel('Angular step [°]'), 4, 0)
        settingsLayout.addWidget(self.thetaStepEdit, 4, 1)
        settingsLayout.addWidget(QtGui.QLabel('Delta Angle [°]'), 5, 0)
        settingsLayout.addWidget(self.deltaAngleEdit, 5, 1)
        settingsLayout.addWidget(powLabel, 6, 0)
        settingsLayout.addWidget(self.sinPowerEdit, 6, 1)
        settingsLayout.addWidget(wvlenLabel, 7, 0)
        settingsLayout.addWidget(self.wvlenEdit, 7, 1)
        settingsLayout.addWidget(self.corrButton, 8, 0, 1, 2)
        settingsFrame.setFixedHeight(260)

        buttonsLayout = QtGui.QGridLayout()
        self.buttonWidget.setLayout(buttonsLayout)
        buttonsLayout.addWidget(loadFrame, 0, 0)
        buttonsLayout.addWidget(settingsFrame, 1, 0)

        self.loadSTORMButton.clicked.connect(self.loadSTORM)
        self.loadSTEDButton.clicked.connect(self.loadSTED)

        self.corrButton.clicked.connect(self.ringFinder)

        # Load sample STED image
        self.loadSTED(os.path.join(os.getcwd(), 'labnanofisica', 'ringfinder',
                                   'spectrin1.tif'))

    def loadSTED(self, filename=None):
        self.loadImage(np.float(self.STEDPxEdit.text()),
                       filename=filename)

    def loadSTORM(self, filename=None):
        # The STORM image has black borders because it's not possible to
        # localize molecules near the edge of the widefield image.
        # Therefore we need to crop those borders before running the analysis.
        nExcluded = np.float(self.excludedEdit.text())
        mag = np.float(self.magnificationEdit.text())
        self.loadImage(np.float(self.STORMPxEdit.text()), crop=nExcluded*mag,
                       filename=filename)

    def loadImage(self, pxSize, crop=0, filename=None):

        self.pxSize = pxSize

        self.inputVb.clear()

        if not(isinstance(filename, str)):
            self.filename = utils.getFilename("Load image",
                                              [('Tiff file', '.tif')])
        else:
            self.filename = filename

        self.inputData = np.array(Image.open(self.filename))
        self.shape = self.inputData.shape
        self.inputData = self.inputData[crop:self.shape[0] - crop,
                                        crop:self.shape[1] - crop]
        self.shape = self.inputData.shape
        self.inputVb.addItem(self.inputImgItem)
        self.inputImgItem.setImage(self.inputData)

        # We need 1um n-sized subimages
        subimgPxSize = 1000/self.pxSize
        self.n = (np.array(self.shape)/subimgPxSize).astype(int)
        self.grid = tools.Grid(self.inputVb, self.shape, self.n)

        self.inputVb.setLimits(xMin=-0.05*self.shape[0],
                               xMax=1.05*self.shape[0], minXRange=4,
                               yMin=-0.05*self.shape[1],
                               yMax=1.05*self.shape[1], minYRange=4)

        self.dataMean = np.mean(self.inputData)
        self.dataStd = np.std(self.inputData)

    def ringFinder(self):
        """RingFinder handles the input data, and then evaluates every subimg
        using the given algorithm which decides if there are rings or not.
        Subsequently gives the output data and plots it"""

        # initialize variables
        self.outputImg.clear()
        self.outputResult.clear()

        # m is such that the image has m x m subimages
        m = self.subimgNum

        # shape the data into the subimg that we need for the analysis
        mean = np.mean(self.inputData)
        std = np.std(self.inputData)
        self.blocksInput = tools.blockshaped(self.inputData,
                                             self.inputData.shape[0]/m,
                                             self.inputData.shape[0]/m)

        # initialize the matrix for storing the ring detection in each subimg
        M = np.zeros(m**2, dtype=bool)
        nBlocks = np.shape(self.blocksInput)[0]
        self.localCorr = np.zeros(nBlocks)

        # for every subimg, we apply the correlation method for ring finding
        minLen = np.float(self.lineLengthEdit.text())/self.pxSize
        wvlen = np.float(self.wvlenEdit.text())/self.pxSize
        sigma = np.float(self.sigmaEdit.text())/self.pxSize
        for i in np.arange(nBlocks):

            block = self.blocksInput[i, :, :]

            # First discrimination for signal level.
            if np.any(block > mean + 3*std):
                args = [np.float(self.corr2thrEdit.text()), sigma, minLen,
                        np.float(self.thetaStepEdit.text()),
                        np.float(self.deltaAngleEdit.text()), wvlen,
                        np.float(self.sinPowerEdit.text())]
                output = tools.corrMethod(block, *args)
                angle, corrTheta, corrMax, theta, phase, rings = output

                # Store results
                self.localCorr[i] = corrMax
                M[i] = rings

            else:
                self.localCorr[i] = 0
                M[i] = False

        # code for visualization of the output
        M1 = M.reshape(m, m)
        self.outputData = np.kron(M1, np.ones((self.inputData.shape[0]/m,
                                               self.inputData.shape[0]/m)))
        self.outputImg.setImage(self.inputData)
        self.outputResult.setImage(self.outputData)
        self.outputResult.setZValue(10)  # make sure this image is on top
        self.outputResult.setOpacity(0.5)

#        self.setGrid(self.outputVb,n=10)
#
#        # plot histogram of the correlation values
#        plt.figure(0)
#        plt.subplot()
#        plt.hist(self.localCorr)
#        plt.title("Correlations Histogram")
#        plt.xlabel("Value")
#        plt.ylabel("Frequency")
#
        plt.figure()
        data = np.array(self.localCorr).reshape(m, m)
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


if __name__ == '__main__':

    app = QtGui.QApplication([])

    win = Gollum()
    win.show()
    app.exec_()
