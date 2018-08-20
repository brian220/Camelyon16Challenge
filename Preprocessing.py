#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ICIAR2018 - Grand Challenge on Breast Cancer Histology Images
https://iciar2018-challenge.grand-challenge.org/home/
"""
import sys
sys.path.append("../FindRoiBbox")
sys.path.append("../ExtractPatches")
from FindRoiBbox import FindRoiBbox
from ExtractPatches import ExtractPatches
import openslide
from openslide import open_slide # http://openslide.org/api/python/
import numpy as np
import os

slideDir = r'C:\Users\nctu\Desktop\Caymelon16Challenge\slide\normal_001.tif'
patchSaveDir = r'C:\Users\nctu\Desktop\Caymelon16Challenge\patch'

class Preprocessing(object):
  def __init__(self):
      pass
  def preprocessing(self):
    wsiSlide, rgbImage, levelUsed = self.scanFile()

    frb = FindRoiBbox()
    openImage, boundingBoxes = frb.findRoiBbox(rgbImage)

    ep = ExtractPatches()
    patchPrefix = "slide"
    patchIndex = 0
    currentIndex = ep.extractNegativePatchesFromNormal(wsiSlide, openImage, levelUsed,
                                                       boundingBoxes, patchSaveDir, patchPrefix,
                                                       patchIndex)

  def scanFile(self):
    wsiSlide = openslide.OpenSlide(slideDir)
    levelUsed = wsiSlide.level_count - 1
    rgbImage = np.array(wsiSlide.read_region((0, 0), levelUsed, wsiSlide.level_dimensions[levelUsed]))
    return wsiSlide, rgbImage, levelUsed

test = Preprocessing()
test.preprocessing()
