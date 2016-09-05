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
from pyqtgraph.Qt import QtGui

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
        self.mainLayout.addWidget(self.inputWidget, 0, 0, 2, 1)
        self.mainLayout.addWidget(self.outputWidget, 0, 1, 2, 1)
        self.mainLayout.addWidget(self.buttonWidget, 0, 2, 1, 1)

        self.inputImgItem = pg.ImageItem()
        self.inputVb = self.inputWidget.addViewBox(col=0, row=0)
        self.inputVb.setAspectLocked(True)
        self.inputVb.addItem(self.inputImgItem)

        inputImgHist = pg.HistogramLUTItem()
        inputImgHist.gradient.loadPreset('thermal')
        inputImgHist.setImageItem(self.inputImgItem)
        inputImgHist.vb.setLimits(yMin=0, yMax=20000)
        self.inputWidget.addItem(inputImgHist)

        path = os.path.join(os.getcwd(), 'labnanofisica', 'ringfinder',
                            'spectrin1.tif')
        imInput = Image.open(path)
        self.inputData = np.array(imInput)
        self.inputImgItem.setImage(self.inputData)
        self.inputDataSize = np.size(self.inputData[0])

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

        buttonsLayout = QtGui.QGridLayout()
        self.buttonWidget.setLayout(buttonsLayout)

        tools.setGrid(self.inputVb, self.inputData)

        self.FFT2Button = QtGui.QPushButton('FFT 2D')
        self.corrButton = QtGui.QPushButton('Correlation')
        self.deltaAngleEdit = QtGui.QLineEdit('30')
        self.thetaStepEdit = QtGui.QLineEdit('3')
        self.pointsButton = QtGui.QPushButton('Points')
        self.fftThrEdit = QtGui.QLineEdit('0.6')
        self.pointsThrEdit = QtGui.QLineEdit('0.6')
#        self.subimgNumEdit = QtGui.QLineEdit('10')
        self.sigmaEdit = QtGui.QLineEdit('60')
        self.lineLengthEdit = QtGui.QLineEdit('0.2')
        self.corr2thrEdit = QtGui.QLineEdit('0.1')
        self.sinPowerEdit = QtGui.QLineEdit('2')
        self.wvlenEdit = QtGui.QLineEdit('180')

        buttonsLayout.addWidget(QtGui.QLabel('STORM pixel (nm)'), 0, 0)
        self.STORMPxEdit = QtGui.QLineEdit('133')
        buttonsLayout.addWidget(self.STORMPxEdit, 0, 1)
        self.loadSTORMButton = QtGui.QPushButton('Load STORM Image')
        buttonsLayout.addWidget(self.loadSTORMButton, 0, 2)

        buttonsLayout.addWidget(QtGui.QLabel('STED pixel (nm)'), 1, 0)
        self.STEDPxEdit = QtGui.QLineEdit('20')
        buttonsLayout.addWidget(self.STEDPxEdit, 1, 1)
        self.loadSTEDButton = QtGui.QPushButton('Load STED Image')
        buttonsLayout.addWidget(self.loadSTEDButton, 1, 2)

        buttonsLayout.addWidget(self.corrButton, 2, 0, 1, 2)
        buttonsLayout.addWidget(QtGui.QLabel('Correlation threshold'), 3, 0)
        buttonsLayout.addWidget(self.corr2thrEdit, 3, 1)
        buttonsLayout.addWidget(QtGui.QLabel('Angular step (°)'), 4, 0)
        buttonsLayout.addWidget(self.thetaStepEdit, 4, 1)
        buttonsLayout.addWidget(QtGui.QLabel('Delta Angle (°)'), 5, 0)
        buttonsLayout.addWidget(self.deltaAngleEdit, 5, 1)
        buttonsLayout.addWidget(QtGui.QLabel('Sinusoidal pattern power'), 6, 0)
        buttonsLayout.addWidget(self.sinPowerEdit, 6, 1)
        wvlenLabel = QtGui.QLabel('wvlen of corr pattern (nm)')
        buttonsLayout.addWidget(wvlenLabel, 7, 0)
        buttonsLayout.addWidget(self.wvlenEdit, 7, 1)
        dirParLabel = QtGui.QLabel('Get direction parameters')
        buttonsLayout.addWidget(dirParLabel, 8, 0, 1, 2)
        buttonsLayout.addWidget(QtGui.QLabel('Sigma of gaussian filter'), 9, 0)
        buttonsLayout.addWidget(self.sigmaEdit, 9, 1)
        buttonsLayout.addWidget(QtGui.QLabel('Direction lines length'), 10, 0)
        buttonsLayout.addWidget(self.lineLengthEdit, 10, 1)

        self.loadSTORMButton.clicked.connect(self.loadSTORM)
        self.loadSTEDButton.clicked.connect(self.loadSTED)

        self.corrButton.clicked.connect(self.ringFinder)

    def loadSTED(self):
        self.loadImage(np.float(self.STEDPxEdit.text()))

    def loadSTORM(self):
        self.loadImage(np.float(self.STORMPxEdit.text()))

    def loadImage(self, pxSize, crop=0):

        self.inputVb.clear()
        subimgPxSize = 1000/pxSize
        self.filename = utils.getFilename("Load image",
                                          [('Tiff file', '.tif')])
        self.loadedImage = Image.open(self.filename)
        self.inputData = np.array(self.loadedImage)
        self.inputVb.addItem(self.inputImgItem)
        self.inputImgItem.setImage(self.inputData)
        self.inputDataSize = np.size(self.inputData[0])
        n = np.int(np.shape(self.inputData)[0]/subimgPxSize)
        tools.setGrid(self.inputVb, self.inputData, n)

        self.inputVb.setLimits(xMin=-0.5, xMax=2*self.inputData.shape[0] - 0.5,
                               minXRange=4, yMin=-0.5,
                               yMax=2*self.inputData.shape[1] - 0.5,
                               minYRange=4)

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
        self.blocksInput = tools.blockshaped(self.inputData,
                                             self.inputDataSize/m,
                                             self.inputDataSize/m)

        # initialize the matrix for storing the ring detection in each subimg
        M = np.zeros(m**2, dtype=bool)
        nBlocks = np.shape(self.blocksInput)[0]
        self.localCorr = np.zeros(nBlocks)
        for i in np.arange(nBlocks):

            # for every subimg, we apply the correlation method for
            # ring finding
            args = [np.float(self.corr2thrEdit.text()),
                    np.float(self.sigmaEdit.text()),
                    np.float(self.pxSizeEdit.text()),
                    np.float(self.lineLengthEdit.text()),
                    np.float(self.thetaStepEdit.text()),
                    np.float(self.deltaAngleEdit.text()),
                    np.float(self.wvlenEdit.text()),
                    np.float(self.sinPowerEdit.text())]
            output = tools.corrMethod(self.blocksInput[i, :, :], *args)
            angle, corrTheta, corrMax, theta, phase, rings = output

            # Store results
            self.localCorr[i] = corrMax
            M[i] = rings

        # code for visualization of the output
        M1 = M.reshape(m, m)
        self.outputData = np.kron(M1, np.ones((self.inputDataSize/m,
                                               self.inputDataSize/m)))
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
