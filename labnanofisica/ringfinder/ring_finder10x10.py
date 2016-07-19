# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 12:25:40 2016

@author: Cibion
"""

from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
#from skimage import img_as_float


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from neurosimulations import simAxon

#from scipy import ndimage as ndi
#import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
#from pyqtgraph.dockarea import Dock, DockArea
#import pyqtgraph.ptime as ptime
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
        
    #Calculating mean values
    AM=np.mean(a)
    BM=np.mean(b)  
    
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
    
def cosTheta(a,b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    
    cosTheta = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
    
    return cosTheta    
    
def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))
        
        
class RingAnalizer10x10(QtGui.QMainWindow):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        self.setWindowTitle('RingAnalizer')
        
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
        self.inputVb = self.inputWidget.addViewBox(col=0,row=0)
        self.inputVb.addItem(self.inputImg)
        
        inputImgHist = pg.HistogramLUTItem()
        inputImgHist.gradient.loadPreset('thermal')
        inputImgHist.setImageItem(self.inputImg)
        self.inputWidget.addItem(inputImgHist)
        
        imInput = Image.open('spectrin1.tif')
        self.inputData = np.array(imInput)
        self.inputImg.setImage(self.inputData)
        self.inputDataSize = np.size(self.inputData[0])


        
        self.outputImg = pg.ImageItem()
        self.outputVb = self.outputWidget.addViewBox(col=0,row=0)
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
        
        self.FFT2Button = QtGui.QPushButton('FFT 2D')
        self.corrButton = QtGui.QPushButton('Correlation')
        self.pointsButton = QtGui.QPushButton('Points')
        self.loadimageButton = QtGui.QPushButton('Load Image')
        self.fftThrEdit = QtGui.QLineEdit('0.6')
        self.pointsThrEdit = QtGui.QLineEdit('0.6')
        self.subimgNumEdit = QtGui.QLineEdit('10')
        
        self.buttonsLayout.addWidget(self.FFT2Button, 0, 0, 1, 1)
#        self.buttonsLayout.addWidget(self.fftThrEdit, 1, 0, 1, 1)
        self.buttonsLayout.addWidget(self.corrButton, 2, 0, 1, 1)
        self.buttonsLayout.addWidget(self.pointsButton, 3, 0, 1, 1)
#        self.buttonsLayout.addWidget(self.pointsThrEdit, 4, 0, 1, 1)
        self.buttonsLayout.addWidget(self.loadimageButton, 5, 0, 1, 1)
        self.buttonsLayout.addWidget(self.subimgNumEdit, 6, 0, 1, 1)
        
        self.loadimageButton.clicked.connect(self.loadImage)
        def rFpoints():
            return self.RingFinder(self.pointsAnalysis)
        self.pointsButton.clicked.connect(rFpoints)
        def rFfft2():
            return self.RingFinder(self.RingFinder(self.FFT2))
        self.FFT2Button.clicked.connect(rFfft2)
        
        
        
    def loadImage(self):
        
        self.filename = getFilename("Load image", [('Tiff file', '.tif')])
        self.loadedImage = Image.open(self.filename)
        self.inputData = np.array(self.loadedImage)
        self.inputImg.setImage(self.inputData)
        self.inputDataSize = np.size(self.inputData[0])
        
    def cropImage(self):
        
        n = np.float(self.subimgNumEdit.text())
        self.blocksInput = blockshaped(self.inputData, self.inputDataSize/n, self.inputDataSize/n)
        
        return self.blocksInput
        
    def RingFinder(self, algorithm):
        
        self.outputImg.clear()
        self.outputResult.clear()
        m = np.float(self.subimgNumEdit.text())
        FFT2input = self.cropImage()
#        self.subimg = np.float(self.fftThrEdit.text())
#        self.outputImg.setImage(FFT2input[self.subimg])
#        self.outputData = np.zeros(np.shape(self.inputData))
        self.blocksInput = blockshaped(self.inputData, self.inputDataSize/m, self.inputDataSize/m)
#        print(self.outputData)
        M = np.zeros(m**2)
        intTot = np.sum(self.inputData)
#        print(intTot)
#        print(m)
#        print(np.shape(self.blocksInput))
#        print(np.arange(0,np.shape(FFT2input)[0]))
#        print(np.size(np.arange(0,np.shape(FFT2input)[0])))
#        
        for i in np.arange(0,np.shape(FFT2input)[0]):
            
        # algorithm for FFT 2D
            
            if np.sum(self.blocksInput[i,:,:]) > intTot/m**2 and algorithm(self.blocksInput[i,:,:]) :
                M[i] = 1
                
#        print(M)
        M1 = M.reshape(m,m)
#        print(np.shape(M1))
        self.outputData = ndi.interpolation.zoom(M1,500/m,order=0)
#        print(np.shape(self.outputData))
#        self.outputImg.setImage(M1)
        self.outputImg.setImage(self.inputData)
        self.outputResult.setImage(self.outputData)
        self.outputResult.setZValue(10)  # make sure this image is on top
        self.outputResult.setOpacity(0.5)
        
        
    def FFT2(self, data):        
        pass
#        self.fft2.clear()
##        self.selected = self.roi.getArrayRegion(self.data, self.bigimg)
##        self.subimgSize = np.shape(self.selected)[0]
#        fft2output = np.log10(np.abs(np.real(np.fft.fftshift(np.fft.fft2(self.selected)))))        
##        i,j = np.unravel_index(fft2output.argmax(), fft2output.shape)
##        fft2output[i,j] = 0
#        
#        self.fft2.addItem(self.FFT2img)
#        self.FFT2img.setImage(fft2output)
#        self.fftThr = np.float(self.main.fftThrEdit.text())
#        coordinates = peak_local_max(fft2output,min_distance=4,threshold_rel=self.fftThr)
#        print(self.fftThr)
#        print(coordinates)
#        
#        A = self.subimgSize
#        rminX = A*(14/100)*np.cos(np.linspace(0, 2*np.pi, 1000))+A/2+1
#        rminY = A*(14/100)*np.sin(np.linspace(0, 2*np.pi, 1000))+A/2+1
#        rmaxX = A*(9/100)*np.cos(np.linspace(0, 2*np.pi, 1000))+A/2+1
#        rmaxY = A*(9/100)*np.sin(np.linspace(0, 2*np.pi, 1000))+A/2+1
#        
#        self.fft2.plot(rminX,rminY, pen=(0,102,204))
#        self.fft2.plot(rmaxX,rmaxY, pen=(0,102,204))
#        self.fft2.plot(coordinates[:, 0],coordinates[:, 1], pen=None, symbolBrush=(0,102,204), symbolPen='w')
##       self.fft2.plot(maxX,maxY, pen=None, symbolBrush=(0,102,204), symbolPen='w')
#        self.fft2.setAspectLocked(True)
#       
#    def corrAnalysis(self):        
#        
#        self.pCorr.clear()
#        
#        self.n = 1
#        self.theta = np.arange(0,180,self.n)
#        
#        self.thetaSteps = np.arange(0,(180/self.n),1)
#        self.phaseSteps = np.arange(0,21,1)
#
#        self.R = np.zeros(np.size(self.thetaSteps))
#        self.R2 = np.zeros(np.size(self.phaseSteps)) 
#        self.R22 = np.zeros(np.size(self.thetaSteps))
#        self.R3 = np.zeros(np.size(self.phaseSteps))
#        self.R33 = np.zeros(np.size(self.thetaSteps))
#        self.R22ph = np.zeros(np.size(self.thetaSteps))
#        
#        wvlen = 9
#        
#        # for now we correlate with the full sin2D pattern
#        
#        for i in self.thetaSteps:
#            for p in self.phaseSteps:
#                self.axonTheta = simAxon(self.subimgSize, wvlen, self.n*i, p*.025, a=0, b=1).simAxon;
#
#                r = corr2(self.selected, self.axonTheta)
#                self.R2[p] = r
#            self.R22[i-1] = np.max(self.R2)
#            self.R22ph[i-1] = np.argmax(self.R2)     #save the phase that maximizes correlation
#            
#        self.thetaMax = self.theta[np.argmax(self.R22)]
#        self.phaseMax = self.R22ph[np.argmax(self.R22)]
#        
#        
#        self.bestAxon = simAxon(self.subimgSize, wvlen, self.thetaMax, self.phaseMax, a=0, b=1).simAxon        
#        
#        self.img1.setImage(self.bestAxon)
#        self.img2.setImage(self.selected)
#        
#        pen = pg.mkPen(color=(0,255,100), width=2, style=QtCore.Qt.SolidLine, antialias = False)
#        self.pCorr.plot(self.theta, self.R22, pen=pen)
#        self.pCorr.showGrid(x=True, y=True)
#               
    def pointsAnalysis(self, data):
        
        self.pointsThr = .65
        points = peak_local_max(data,min_distance=3,threshold_rel=self.pointsThr)
        
        dmin = 8
        dmax = 11

        D = []
#        print(np.arange(np.shape(points)[0]))
        
        # look up every point
        for i in np.arange(0,np.shape(points)[0]-1):
            # calculate the distance of every point to the others
            for j in np.arange(i+1,np.shape(points)[0]):
                d1 = dist(points[i],points[j])
                # if there are two points at the right distance then
                if dmin < d1 < dmax:
                    for k in np.arange(0,np.shape(points)[0]-1):
                        # check the distance between the last point and the other points in the list
                        if k != i & k != j:
                            d2 = dist(points[j],points[k])

                        else:
                            d2 = 0
                        
                        # calculate the angle between vector i-j and j-k with i, j, k points
                        v1 = points[i]-points[j]
                        v2 = points[j]-points[k]
                        t = cosTheta(v1,v2)      
                        
                        # if point k is at right distance from point j and the angle is flat enough
                        if dmin < d2 < dmax and np.abs(t) > 0.9 :
                            # save the three points and plot the connections
                            D.append([points[i],points[j],points[k]])
#                            self.pointsPlot.plot([points[i][0],points[j][0],points[k][0]],
#                                                 [points[i][1],points[j][1],points[k][1]], 
#                                                 pen=pen, symbolBrush=(0,204,122), symbolPen='w')
                        else:
                            pass

        if len(D)>0 :
            return 1
        else:
            return 0
            
        return 1
        
if __name__ == '__main__':

    app = QtGui.QApplication([])
    
    win = RingAnalizer10x10()
    win.show()
    app.exec_()
