# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:51:00 2016

@author: Cibion
"""

import numpy as np
import matplotlib.pyplot as plt


class sin2D:
    def __init__(self, imSize=100, wvlen=10, theta=15, phase=.25, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.imSize = imSize                           # image size: n X n
        self.wvlen = wvlen                             # wavelength (number of pixels per cycle)
        self.theta = theta                              # grating orientation
        self.phase = phase                            # phase (0 -> 1)

        self.pi = np.pi                      
        
        self.X = np.arange(1, self.imSize+1)                           # X is a vector from 1 to imageSize
        self.X0 = (self.X / self.imSize) - .5                 # rescale X -> -.5 to .5                              
        
        self.freq = self.imSize/self.wvlen                    # compute frequency from wavelength
        self.phaseRad = (self.phase * 2* self.pi)             # convert to radians: 0 -> 2*pi
        
        [self.Xm, self.Ym] = np.meshgrid(self.X0, self.X0)             # 2D matrices

        self.thetaRad = (self.theta / 360) * 2*self.pi        # convert theta (orientation) to radians
        self.Xt = self.Xm * np.cos(self.thetaRad)                # compute proportion of Xm for given orientation 
        self.Yt = self.Ym * np.sin(self.thetaRad)                # compute proportion of Ym for given orientation
        self.XYt = np.array(self.Xt + self.Yt)                  # sum X and Y components
        self.XYf = self.XYt * self.freq * 2*self.pi                # convert to radians and scale by frequency

        self.sin2d = np.sin(self.XYf + self.phaseRad)                   # make 2D sinewave
        

#        plt.figure()
#        plt.imshow(self.grating, cmap='gray')                    # display
        
class simAxon(sin2D):
        def __init__(self, imSize=50, wvlen=10, theta=15, phase=.25, a=0, b=2, *args, **kwargs):

            super().__init__(*args, **kwargs)
            
            self.imSize = imSize                           # image size: n X n
            self.wvlen = wvlen                             # wavelength (number of pixels per cycle)
            self.theta = theta                              # grating orientation
            self.phase = phase                            # phase (0 -> 1)            
            
            if b%2==0:
                self.grating2 = sin2D(self.imSize, 2*self.wvlen, 90 - self.theta, self.phase).sin2d**b;   # sin2D.sin2d squared in order to always get positive values
                self.mask = sin2D(self.imSize,2*self.imSize,-self.theta+90,phase).sin2d**a
            else: 
                self.grating2 = sin2D(self.imSize, self.wvlen, 90 - self.theta, self.phase).sin2d**b;   # sin2D.sin2d squared in order to always get positive values
                self.mask = sin2D(self.imSize,2*self.imSize,-self.theta+90,phase).sin2d**a
                
            self.simAxon = self.grating2*(self.mask)        # make simulated axon
        
    


        