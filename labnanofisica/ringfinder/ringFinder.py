# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 12:25:40 2016

@author: Luciano Masullo, Federico Barabas
"""

import os
import numpy as np

from scipy import ndimage as ndi
from skimage import filters
from skimage.transform import probabilistic_hough_line
from skimage.filters import threshold_otsu

from PIL import Image
import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from labnanofisica.ringfinder.neurosimulations import simAxon
import labnanofisica.utils as utils
import labnanofisica.ringfinder.tools as tools


class RingAnalizer(QtGui.QMainWindow):

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

        self.inputImg = pg.ImageItem()
        self.inputVb = self.inputWidget.addViewBox(col=0, row=0)
        self.inputVb.setAspectLocked(True)
        self.inputVb.addItem(self.inputImg)

        inputImgHist = pg.HistogramLUTItem()
        inputImgHist.gradient.loadPreset('thermal')
        inputImgHist.setImageItem(self.inputImg)
        self.inputWidget.addItem(inputImgHist)

        path = os.path.join(os.getcwd(),
                            r'labnanofisica\ringfinder\spectrin1.tif')
        imInput = Image.open(path)
        self.inputData = np.array(imInput)
        self.inputImg.setImage(self.inputData)
        self.inputDataSize = np.size(self.inputData[0])

        self.outputImg = pg.ImageItem()
        self.outputVb = self.outputWidget.addViewBox(col=0, row=0)
        self.outputVb.setAspectLocked(True)
        self.outputVb.addItem(self.outputImg)

        outputImgHist = pg.HistogramLUTItem()
        outputImgHist.gradient.loadPreset('thermal')
        outputImgHist.setImageItem(self.outputImg)
        self.outputWidget.addItem(outputImgHist)

#        imOutput = Image.open('spectrin1.tif')
#        self.outputData = np.array(imOutput)
#        self.outputImg.setImage(np.rot90(self.outputData))

        self.outputResult = pg.ImageItem()
        self.outputVb.addItem(self.outputResult)

        self.buttonsLayout = QtGui.QGridLayout()
        self.buttonWidget.setLayout(self.buttonsLayout)

        self.setGrid(self.inputVb, n=10)

        self.FFT2Button = QtGui.QPushButton('FFT 2D')
        self.corrButton = QtGui.QPushButton('Correlation')
        self.deltaAngleEdit = QtGui.QLineEdit('30')
        self.thetaStepEdit = QtGui.QLineEdit('3')
        self.pointsButton = QtGui.QPushButton('Points')
        self.loadimageButton = QtGui.QPushButton('Load Image')
        self.fftThrEdit = QtGui.QLineEdit('0.6')
        self.pointsThrEdit = QtGui.QLineEdit('0.6')
#        self.subimgNumEdit = QtGui.QLineEdit('10')
        self.sigmaEdit = QtGui.QLineEdit('60')
        self.lineLengthEdit = QtGui.QLineEdit('0.2')
        self.pxSizeEdit = QtGui.QLineEdit('20')
        self.corr2thrEdit = QtGui.QLineEdit('0.1')
        self.sinPowerEdit = QtGui.QLineEdit('2')
        self.wvlenEdit = QtGui.QLineEdit('180')

        self.corr2thrLabel = QtGui.QLabel('Correlation threshold')
        self.thetaStepLabel = QtGui.QLabel('Angular step (°)')
        self.deltaAngleLabel = QtGui.QLabel('Delta Angle (°)')
        self.sinPowerLabel = QtGui.QLabel('Sinusoidal pattern power')
        self.sigmaLabel = QtGui.QLabel('Sigma of gaussian filter')
        self.pxSizeLabel = QtGui.QLabel('Pixel Size (nm)')
        self.wvlenLabel = QtGui.QLabel('wvlen of corr pattern (nm)')
        self.lineLengthLabel = QtGui.QLabel('Direction lines length')
        self.getDirParam = QtGui.QLabel('Get direction parameters')
#        self.buttonsLayout.addWidget(self.FFT2Button, 0, 0, 1, 2)
#        self.buttonsLayout.addWidget(self.pointsButton, 1, 0, 1, 2)
        self.buttonsLayout.addWidget(self.corrButton, 0, 0, 1, 1)
        self.buttonsLayout.addWidget(self.corr2thrLabel, 1, 0, 1, 1)
        self.buttonsLayout.addWidget(self.corr2thrEdit, 1, 1, 1, 1)
        self.buttonsLayout.addWidget(self.thetaStepLabel, 2, 0, 1, 1)
        self.buttonsLayout.addWidget(self.thetaStepEdit, 2, 1, 1, 1)
        self.buttonsLayout.addWidget(self.deltaAngleLabel, 3, 0, 1, 1)
        self.buttonsLayout.addWidget(self.deltaAngleEdit, 3, 1, 1, 1)
        self.buttonsLayout.addWidget(self.sinPowerLabel, 4, 0, 1, 1)
        self.buttonsLayout.addWidget(self.sinPowerEdit, 4, 1, 1, 1)
        self.buttonsLayout.addWidget(self.wvlenLabel, 5, 0, 1, 1)
        self.buttonsLayout.addWidget(self.wvlenEdit, 5, 1, 1, 1)
        self.buttonsLayout.addWidget(self.getDirParam, 6, 0, 1, 2)
        self.buttonsLayout.addWidget(self.sigmaLabel, 7, 0, 1, 1)
        self.buttonsLayout.addWidget(self.sigmaEdit, 7, 1, 1, 1)
        self.buttonsLayout.addWidget(self.lineLengthLabel, 8, 0, 1, 1)
        self.buttonsLayout.addWidget(self.lineLengthEdit, 8, 1, 1, 1)

        self.buttonsLayout.addWidget(self.loadimageButton, 9, 0, 1, 1)
#        self.buttonsLayout.addWidget(self.subimgNumEdit, 7, 0, 1, 1)
        self.buttonsLayout.addWidget(self.pxSizeLabel, 10, 0, 1, 1)
        self.buttonsLayout.addWidget(self.pxSizeEdit, 10, 1, 1, 1)

        self.loadimageButton.clicked.connect(self.loadImage)

        def rFpoints():
            return self.RingFinder(tools.points)

        self.pointsButton.clicked.connect(rFpoints)

        def rFfft2():
            return self.RingFinder(tools.FFT2)

        self.FFT2Button.clicked.connect(rFfft2)

        def rFcorr():
            return self.RingFinder(self.corr2)

        self.corrButton.clicked.connect(rFcorr)

    def loadImage(self):

        self.inputVb.clear()
        pxSize = np.float(self.pxSizeEdit.text())
        subimgPxSize = 1000/pxSize
        self.filename = utils.getFilename("Load image",
                                          [('Tiff file', '.tif')])
        self.loadedImage = Image.open(self.filename)
        self.inputData = np.array(self.loadedImage)
        self.inputVb.addItem(self.inputImg)
        self.inputImg.setImage(self.inputData)
        self.inputDataSize = np.size(self.inputData[0])
        n = np.int(np.shape(self.inputData)[0]/subimgPxSize)
        self.setGrid(self.inputVb, n)

    def setGrid(self, image, n):

        pen = QtGui.QPen(QtCore.Qt.yellow, 1, QtCore.Qt.SolidLine)

        xlines = []
        ylines = []

        for i in np.arange(0, n - 1):
            xlines.append(pg.InfiniteLine(pen=pen, angle=0))
            ylines.append(pg.InfiniteLine(pen=pen))

        for i in np.arange(0, n - 1):
            xlines[i].setPos((self.inputDataSize/n)*(i+1))
            ylines[i].setPos((self.inputDataSize/n)*(i+1))
            image.addItem(xlines[i])
            image.addItem(ylines[i])

    def RingFinder(self, algorithm):
        """RingFinder handles the input data, and then evaluates every subimg
        using the given algorithm which decides if there are rings or not.
        Subsequently gives the output data and plots it"""

        # initialize variables
        self.localCorr = []
        a = 0
        self.outputImg.clear()
        self.outputResult.clear()

        # m is such that the image has m x m subimages
        m = self.subimgNum

        # shape the data into the subimg that we need for the analysis
        self.blocksInput = tools.blockshaped(self.inputData,
                                             self.inputDataSize/m,
                                             self.inputDataSize/m)

        # initialize the matrix with the values of 1 and 0 (rings or not)
        M = np.zeros(m**2)

        for i in np.arange(0, np.shape(self.blocksInput)[0]):

            # for every subimg, evaluate it, with the given algorithm
            if algorithm(self.blocksInput[i, :, :])[-1]:
                M[i] = 1
                a = a+1
            else:
                M[i] = 0

        # code for visualization of the output
        M1 = M.reshape(m, m)
        self.outputData = np.kron(M1, np.ones((self.inputDataSize/m,
                                               self.inputDataSize/m)))
        self.outputImg.setImage(self.inputData)
        self.outputResult.setImage(self.outputData)
        self.outputResult.setZValue(10)  # make sure this image is on top
        self.outputResult.setOpacity(0.5)

        print(self.localCorr)
        print(np.size(self.localCorr))
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
                         verticalalignment='center',
                         )

        plt.colorbar(heatmap)

        plt.show()

    def corr2(self, data):
        """Correlates the image with a given sinusoidal pattern"""

        # correlation thr set by the user
        corr2thr = np.float(self.corr2thrEdit.text())

        # mean angle calculated
        meanAngle = self.getDirection(data)
        print('meanAngle is {}'.format(np.around(meanAngle, 1)))

        if meanAngle == 2:
            dataRot = np.rot90(data)
            meanAngle = self.getDirection(dataRot)-90

        if meanAngle == 666:
            self.localCorr.append(0)
            return 0
        else:
            subImgSize = np.shape(data)[0]

            # angular step size set by the user
            n = np.float(self.thetaStepEdit.text())

            # get the max allowed angle from the user
            maxAngle = np.float(self.deltaAngleEdit.text())

            # set the angle range to look for a correlation, 179 is added
            # because of later corrAngle's expansion
            deltaAngle = np.arange(meanAngle-maxAngle,
                                   meanAngle+maxAngle, n, dtype=int)

            print('deltaAngle is {}'.format(deltaAngle))

            thetaSteps = np.arange(0, np.size((deltaAngle-179)/n), 1)
            # phase steps are set to 20, TO DO: explore this parameter
            phaseSteps = np.arange(0, 21, 1)

            corrPhase = np.zeros(np.size(phaseSteps))
            corrAngle = np.zeros(np.size(thetaSteps))

            wvlen_nm = np.float(self.wvlenEdit.text())  # wvlen in nm
            pxSize = np.float(self.pxSizeEdit.text())  # pixel size in nm
            wvlen = wvlen_nm/pxSize  # wvlen in px
            sinPower = np.float(self.sinPowerEdit.text())

            # for now we correlate with the full sin2D pattern

            for i in thetaSteps:
                for p in phaseSteps:
                    # creates simulated axon
                    axonTheta = simAxon(subImgSize, wvlen, deltaAngle[i],
                                        p*.025, a=0, b=sinPower).simAxon
                    # calculates correlation with data
                    c = tools.corr2(data, axonTheta)
                    # saves correlation for the given phase p
                    corrPhase[p] = c
                # saves the correlation for the best p, and given angle i
                corrAngle[i-1] = np.max(corrPhase)

            self.localCorr.append(np.max(corrAngle))

            if np.max(corrAngle) > corr2thr:
                return 1
            else:
                return 0

    def getDirection(self, data):
        """Returns the direction (angle) of the neurite in the image"""

        # dataSize
        dataSize = np.shape(data)[0]

        # gaussian filter to get low resolution image
        sigma = np.float(self.sigmaEdit.text())
        pxSize = np.float(self.pxSizeEdit.text())
        sigma_px = sigma/pxSize
        img = ndi.gaussian_filter(data, sigma_px)

        # TO DO: good cond on intensity
        if np.sum(img) < 10:
            return 666
        else:
            pass

        # binarization of image
        thresh = threshold_otsu(img)
        binary = img > thresh

        # find edges
        edges = filters.sobel(binary)

        # min line length is set by the user
        linLen = np.float(self.lineLengthEdit.text())

        # get directions
        lines = probabilistic_hough_line(edges, threshold=10,
                                         line_length=dataSize*linLen,
                                         line_gap=3)

        # allocate angleArr which will have the angles of the lines
        angleArr = []

        # if cannot find any directions, return 666, code for this case
        if lines == []:
            return 666

        else:
            for line in lines:
                p0, p1 = line

                # get the m coefficient of the lines and the angle
                if p1[1] == p0[1]:
                    angle = 90
                else:
                    m = (p1[0]-p0[0])/(p1[1]-p0[1])
                    angle = (180/np.pi)*np.arctan(m)

                angleArr.append(angle)

            # calculate mean angle and its standard deviation
            print('angleArr is {}'.format(np.around(angleArr, 1)))
            meanAngle = np.mean(angleArr)
            stdAngle = np.std(angleArr)

            # if the std is too high it's probably the case of flat angles,
            # i.e., 181, -2, 0.3, -1, 179, getDirection returns 2, a code
            # for this case.
            # TO DO: find optimal threshold, 40 is arbitrary
            if stdAngle > 40:
                return 2
            else:
                return meanAngle

if __name__ == '__main__':

    app = QtGui.QApplication([])

    win = RingAnalizer()
    win.show()
    app.exec_()
