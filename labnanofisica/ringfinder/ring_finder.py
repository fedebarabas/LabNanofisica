# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 13:41:48 2014

@authors: Luciano Masullo
"""
        
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import data, img_as_float


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from neurosimulations import simAxon
#from maxima import Maxima

#from scipy import ndimage as ndi
#import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu

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
        
class RingAnalizer(QtGui.QMainWindow):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        self.setWindowTitle('RingAnalizer')
        
        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)
       
#        self.imagegui.setFixedSize(800,800)
        
        self.FFT2Button = QtGui.QPushButton('FFT 2D')
        self.corrButton = QtGui.QPushButton('Correlation')
        self.pointsButton = QtGui.QPushButton('Points')
        self.loadimageButton = QtGui.QPushButton('Load Image')
        self.filterImageButton = QtGui.QPushButton('Filter Image')
        self.fftThrEdit = QtGui.QLineEdit('0.6')
        self.pointsThrEdit = QtGui.QLineEdit('0.6')
        self.roiSizeEdit = QtGui.QLineEdit('50')
        # Widgets' layout
        self.layout = QtGui.QGridLayout()
        self.cwidget.setLayout(self.layout)
        self.layout.setColumnStretch(1, 1)
##        self.layout.setColumnMinimumWidth(3, 1000)
#        self.layout.setRowMinimumHeight(0, 500)
#        self.layout.setRowMinimumHeight(2, 40)
#        self.layout.setRowMinimumHeight(3, 30)
        self.imagegui = ImageGUI(self)
        self.layout.addWidget(self.imagegui, 0, 0, 14, 8)
        self.layout.addWidget(self.FFT2Button, 5, 9, 1, 1)
        self.layout.addWidget(self.fftThrEdit, 6, 9, 1, 1)
        self.layout.addWidget(self.corrButton, 2, 9, 1, 1)
        self.layout.addWidget(self.pointsButton, 3, 9, 1, 1)
        self.layout.addWidget(self.pointsThrEdit, 4, 9, 1, 1)
        self.layout.addWidget(self.loadimageButton, 0, 9, 1, 1)
        self.layout.addWidget(self.filterImageButton, 7, 9, 1, 1)
        self.layout.addWidget(self.roiSizeEdit, 1, 9, 1, 1)

        self.roiSizeEdit.textChanged.connect(self.imagegui.updateROI)
        self.loadimageButton.clicked.connect(self.imagegui.loadImage)
        self.FFT2Button.clicked.connect(self.imagegui.FFT2)
        self.corrButton.clicked.connect(self.imagegui.corrAnalysis)
        self.pointsButton.clicked.connect(self.imagegui.pointsAnalysis)
        self.filterImageButton.clicked.connect(self.imagegui.imageFilter)

class ImageGUI(pg.GraphicsLayoutWidget):

    def __init__(self, main, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.main = main
        self.setWindowTitle('ImageGUI')        
        self.subimgSize = 100
        
        self.vb1 = self.addViewBox(row=0, col=0)
        self.bigimg = pg.ImageItem()
        self.vb1.addItem(self.bigimg)
        self.vb1.setAspectLocked(True)
        
        
        # A plot area (ViewBox + axes) for displaying the image

        
        # Item for displaying 10x10 image data
        
        
        # Custom ROI for selecting an image region
        self.roi = pg.ROI([0, 0], [0, 0])
        self.roiSizeX = np.float(self.main.roiSizeEdit.text())
        self.roiSizeY = np.float(self.main.roiSizeEdit.text())
        self.roi.setSize(self.roiSizeX,self.roiSizeY)
        self.roi.addScaleHandle([1, 1], [0, 0], lockAspect=True)
#        self.roi.addScaleHandle([0.5, 0], [0.5, 0.5])
        self.vb1.addItem(self.roi)
        self.roi.setZValue(10)  # make sure ROI is drawn above image
        
        # Contrast/color control
        bigimg_hist = pg.HistogramLUTItem()
        bigimg_hist.gradient.loadPreset('thermal')
        bigimg_hist.setImageItem(self.bigimg)
        self.addItem(bigimg_hist)
        
        #Another plot area for displaying ROI data
        self.nextRow()
        self.vb2 = self.addViewBox(row=1, col=0)
        self.subimg = pg.ImageItem()
        self.vb2.addItem(self.subimg)
        self.vb2.setAspectLocked(True)
#        self.show()
        
        subimg_hist = pg.HistogramLUTItem()
        subimg_hist.gradient.loadPreset('thermal')
        subimg_hist.setImageItem(self.subimg)
        self.addItem(subimg_hist)

        self.im = Image.open('spectrin1.tif')
        self.data = np.array(self.im)
        self.bigimg.setImage(self.data)
        bigimg_hist.setLevels(self.data.min(), self.data.max())
        
        
#        self.vb3 = self.addViewBox(row=1, col=2)
        self.FFT2img = pg.ImageItem()
#        self.vb3.addItem(self.FFT2img)
#        self.vb3.setAspectLocked(True)
        self.FFT2img_hist = pg.HistogramLUTItem(image=self.FFT2img)
        self.FFT2img_hist.gradient.loadPreset('thermal')
        self.addItem(self.FFT2img_hist, row=1, col=3)
            
        self.fft2 = pg.PlotItem(title="FFT 2D")
        self.fft2.addItem(self.FFT2img)
        self.addItem(self.fft2,row=1,col=2)
#        self.fft2.showAxis('bottom', False)
#        self.fft2.showAxis('left', False)
#        self.fft2.showAxis('bottom', False)
#        self.fft2.showAxis('left', False)

        self.pointsImg = pg.ImageItem()
#        self.vb3.addItem(self.FFT2img)
#        self.vb3.setAspectLocked(True)
        self.pointsImg_hist = pg.HistogramLUTItem(image=self.pointsImg)
        self.pointsImg_hist.gradient.loadPreset('thermal')
        self.addItem(self.pointsImg_hist, row=1, col=5)
            
        self.pointsPlot = pg.PlotItem(title="Points")
        self.pointsPlot.addItem(self.pointsImg)
        self.addItem(self.pointsPlot,row=1,col=4)
        


        self.pCorr = pg.PlotItem(title="Correlation")
        self.addItem(self.pCorr,row=0,col=2)
       
       
       # optimal correlation visualization
        
        self.vb4 = self.addViewBox(row=0,col=4)
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
        self.addItem(overlay_hist, row=0, col=5)
      
        self.roi.sigRegionChanged.connect(self.updatePlot)


        
    def loadImage(self):
        self.filename = getFilename("Load image", [('Tiff file', '.tif')])
        self.loadedImage = Image.open(self.filename)
        self.data = np.array(self.loadedImage)
        self.bigimg.setImage(self.data)
        
    # Callbacks for handling user interaction
        
    def updatePlot(self):
        self.selected = self.roi.getArrayRegion(self.data, self.bigimg)
        self.subimg.setImage(self.selected)
        self.subimgSize = np.shape(self.selected)[0] 
        
    def updateROI(self):
        
        self.roiSizeX = np.float(self.main.roiSizeEdit.text())
        self.roiSizeY = np.float(self.main.roiSizeEdit.text())
        self.roi.setSize(self.roiSizeX,self.roiSizeY)

    def FFT2(self):

        # remove previous fft2
        self.fft2.clear()
        
        # calculate new fft2      
        fft2output = np.log10(np.abs(np.real(np.fft.fftshift(np.fft.fft2(self.selected)))))        

        # add new fft2 graph
        self.fft2.addItem(self.FFT2img)
        self.fft2.setAspectLocked(True)
        self.FFT2img.setImage(fft2output)
        
        # set fft2 analysis threshold
        self.fftThr = np.float(self.main.fftThrEdit.text())
        
        # calculate local intensity maxima
        coordinates = peak_local_max(fft2output, min_distance=2, threshold_rel=self.fftThr)
        print(coordinates)

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
        self.fft2.plot(rminX,rminY, pen=(0,102,204))
        self.fft2.plot(rmaxX,rmaxY, pen=(0,102,204))
        
        # plot local maxima        
        self.fft2.plot(coordinates[:, 0], coordinates[:, 1], pen=None, symbolBrush=(0,102,204), symbolPen='w')

        # aux arrays: ringBool is checked to define wether there are rings or not
        ringBool = []
        
        # D saves the distances of local maxima from the centre of the fft2
        D = []
        
        # loop for calculating all the distances d, elements of array D
        
        for i in np.arange(0,np.shape(coordinates)[0]):
            d = dist([A/2,A/2],coordinates[i])
            D.append(dist([A/2,A/2],coordinates[i]))
            if A*(rmin/100) < d < A*(rmax/100):
                ringBool.append(1)
                
#        print(self.ringBool)
                
        # condition for ringBool: all elements d must correspond to periods between 170 and 220 nm
        if np.sum(ringBool) == np.shape(coordinates)[0]-1 and np.sum(ringBool) > 0 :
            print('¡HAY ANILLOS!')
        else:
            print('NO HAY ANILLOS')
            
#        print(self.D)
            
    def pointsAnalysis(self):
        
        self.pointsPlot.clear()
        self.pointsThr = np.float(self.main.pointsThrEdit.text())
        points = peak_local_max(self.selected,min_distance=3,threshold_rel=self.pointsThr)
        self.pointsPlot.addItem(self.pointsImg)
        self.pointsImg.setImage(self.selected)
        self.pointsPlot.plot(points[:, 0],points[:, 1], pen=None, symbolBrush=(0,204,122), symbolPen='w')
        
        dmin = 8
        dmax = 11
        pen = pg.mkPen(color=(0,255,100), width=1, style=QtCore.Qt.SolidLine, antialias = False)
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
                            self.pointsPlot.plot([points[i][0],points[j][0],points[k][0]],
                                                 [points[i][1],points[j][1],points[k][1]], 
                                                 pen=pen, symbolBrush=(0,204,122), symbolPen='w')
                        else:
                            pass

        if len(D)>0 :
            print('¡HAY ANILLOS!')
        else:
            print('NO HAY ANILLOS')
            

#        print(D)
#        print(len(D))
#        print(np.shape(D))

    def corrAnalysis(self):        
        
        self.pCorr.clear()
        
        self.n = 1
        self.theta = np.arange(0,180,self.n)
        
        self.thetaSteps = np.arange(0,(180/self.n),1)
        self.phaseSteps = np.arange(0,21,1)

        self.R = np.zeros(np.size(self.thetaSteps))
        self.R2 = np.zeros(np.size(self.phaseSteps)) 
        self.R22 = np.zeros(np.size(self.thetaSteps))
        self.R3 = np.zeros(np.size(self.phaseSteps))
        self.R33 = np.zeros(np.size(self.thetaSteps))
        self.R22ph = np.zeros(np.size(self.thetaSteps))
        
        wvlen = 9
        
        # for now we correlate with the full sin2D pattern
        
        for i in self.thetaSteps:
            for p in self.phaseSteps:
                self.axonTheta = simAxon(self.subimgSize, wvlen, self.n*i, p*.025, a=0, b=1).simAxon;

                r = corr2(self.selected, self.axonTheta)
                self.R2[p] = r
            self.R22[i-1] = np.max(self.R2)
            self.R22ph[i-1] = np.argmax(self.R2)     #save the phase that maximizes correlation
            
        self.thetaMax = self.theta[np.argmax(self.R22)]
        self.phaseMax = self.R22ph[np.argmax(self.R22)]
        
        
        self.bestAxon = simAxon(self.subimgSize, wvlen, self.thetaMax, self.phaseMax, a=0, b=1).simAxon        
        
        self.img1.setImage(self.bestAxon)
        self.img2.setImage(self.selected)
        
        pen = pg.mkPen(color=(0,255,100), width=2, style=QtCore.Qt.SolidLine, antialias = False)
        self.pCorr.plot(self.theta, self.R22, pen=pen)
        self.pCorr.showGrid(x=True, y=True)
        
    def imageFilter(self):
        
        sigma = 4
        img = ndi.gaussian_filter(self.data,sigma)
    
        thresh = threshold_otsu(img)
        binary = img > thresh  
        
        self.data = self.data*binary
        self.bigimg.setImage(self.data)
        
               
if __name__ == '__main__':

    app = QtGui.QApplication([])
    
    win = RingAnalizer()
    win.show()
    app.exec_()
