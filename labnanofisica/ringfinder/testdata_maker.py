# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 14:37:26 2016

@author: Federico Barabas
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff

import labnanofisica.utils as utils
import labnanofisica.ringfinder.tools as tools


def loadData(folder, ax, subimgPxSize, technique, mag=None):
    """
    Plot loaded image, blocks grid and numbers
    """

    # Load image
    filename = utils.getFilename('Load image',
                                 [('Tiff file', '.tif')], folder)
    newFolder = os.path.split(filename)[0]

    inputData = tiff.imread(filename)
    dataShape = inputData.shape

    if technique == 'STORM':
        crop = 3*mag
        bound = (np.array(dataShape) - crop).astype(np.int)
        inputData = inputData[crop:bound[0], crop:bound[1]]
        dataShape = inputData.shape

    n = (np.array(dataShape)/subimgPxSize).astype(int)

    plotWithGrid(inputData, ax, subimgPxSize)

    # Segment image in blocks
    nblocks = np.array(dataShape)/n
    blocks = tools.blockshaped(inputData, *nblocks)

    return newFolder, blocks, dataShape


def selectBlocks(needRings, needNoRings):

    text = 'We need {} rings and {} no rings blocks'
    print(text.format(needRings, needNoRings))

    listRings = input("Select nice ring blocks (i.e. '1-3-11-20') ")
    listRings = [int(s) for s in listRings.split('-')]

    listNoRings = input("Select non-ring (but still neuron) blocks "
                        "(i.e. '2-7-14-24') ")
    listNoRings = [int(s) for s in listNoRings.split('-')]

    return listRings, listNoRings


def plotWithGrid(data, ax, subimgPxSize):

    dataShape = data.shape
    n = (np.array(dataShape)/subimgPxSize).astype(int)

    plt.imshow(data, interpolation='None')
    plt.colorbar()
    xticks = np.linspace(0, dataShape[0], n[0], endpoint=False)
    xticks = xticks[1:]
    dx = xticks[1] - xticks[0]
    yticks = np.linspace(0, dataShape[1], n[1], endpoint=False)
    yticks = yticks[1:]
    dy = yticks[1] - yticks[0]
    ax.set_xticks(xticks, minor=False)
    ax.set_yticks(yticks, minor=False)
    ax.xaxis.grid(True, which='major', color='w', linewidth=1)
    ax.yaxis.grid(True, which='major', color='w', linewidth=1)
    for y in np.arange(n[1]):
        for x in np.arange(n[1]):
            plt.text((x + 0.5)*dx, (y + 0.5)*dy, '{}'.format(x + 10*y),
                     color='w', horizontalalignment='center',
                     verticalalignment='center')


def buildData(technique, pxSize, mag=None):

    subimgPxSize = 1000/pxSize
    folder = os.getcwd()
    nRings = 0
    nNoRings = 0

    print("Test image creation script started...")
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 18, forward=True)

    try:
        folder, blocks, dataShape = loadData(os.getcwd(), ax, subimgPxSize,
                                             technique, mag)
        plt.show(block=False)

        # this array will be the output
        testData = np.zeros(blocks.shape, dtype=blocks.dtype)
        maxRings = int(0.5*len(blocks))
        maxNoRings = maxRings

        listRings, listNoRings = selectBlocks(maxRings, maxRings)
        lR = len(listRings)
        testData[nRings:nRings + lR] = blocks[listRings]
        nRings += lR
        lNR = len(listNoRings)
        nRBlocks = blocks[listNoRings]
        testData[maxRings + nNoRings:maxRings + nNoRings + lNR] = nRBlocks
        nNoRings += lNR

        keepWorking = input('Keep working? [y/n] ') == 'y'

    except OSError:
        keepWorking = False

    while keepWorking:

        try:
            folder, blocks, dd = loadData(folder, ax, subimgPxSize,
                                          technique, mag)

            listRings, listNoRings = selectBlocks(maxRings - nRings,
                                                  maxNoRings - nNoRings)
            lR = len(listRings)
            testData[nRings:nRings + lR] = blocks[listRings]
            nRings += lR
            lNR = len(listNoRings)
            nRBlocks = blocks[listNoRings]
            testData[maxRings + nNoRings:maxRings + nNoRings + lNR] = nRBlocks
            nNoRings += lNR

            keepWorking = input('Keep working? [y/n] ') == 'y'

        except OSError:
            print('No file selected!')

    testData = tools.unblockshaped(testData, *dataShape)
    plotWithGrid(testData, ax, subimgPxSize)
    plt.show()

    tiff.imsave('testdata.tif', testData)

if __name__ == '__main__':

    buildData('STED', 20)
#    buildData('STORM', 13.3, 10)
