# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 11:05:51 2016

@author: Cibion
"""


import numpy as np

from scipy.special import erf
from scipy.optimize import minimize
from scipy.ndimage import label
from scipy.ndimage.filters import convolve, maximum_filter
from scipy.ndimage.measurements import maximum_position, center_of_mass

import labnanofisica.sm.tools as tools


import warnings
warnings.filterwarnings("error")