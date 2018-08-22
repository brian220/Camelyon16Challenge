"""
ICIAR2018 - Grand Challenge on Breast Cancer Histology Images
https://iciar2018-challenge.grand-challenge.org/home/
"""
import sys
sys.path.append("../RoiOperation")
sys.path.append("../TumorMaskOperation")
sys.path.append("../ExtractPatches")
from RoiOperation import RoiOperation
from TumorMaskOperation import TumorMaskOperation
from ExtractPatches import ExtractPatches
import openslide
from openslide import open_slide # http://openslide.org/api/python/
import cv2
import numpy as np
import os

slideDir = r'C:\Users\nctu\Desktop\Caymelon16Challenge\slide\tumor_005.tif'
xmlDir = r'C:\Users\nctu\Desktop\Caymelon16Challenge\xmlFiles\tumor_005.xml'
patchNormalNegativeSaveDir = r'C:\Users\nctu\Desktop\Caymelon16Challenge\patch\Normal'

patchTumorPositiveSaveDir = r'C:\Users\nctu\Desktop\Caymelon16Challenge\patch\Tumor\positive'
patchTumorNegativeSaveDir = r'C:\Users\nctu\Desktop\Caymelon16Challenge\patch\Tumor\negative'

class NormalPreprocessing(object):
  def __init__(self, slideDir, positivePatchIndex):
    self.slideDir = slideDir
    self.positivePatchIndex = positivePatchIndex
    pass

  def normalPreprocessing(self):
    wsiSlide, rgbImage, levelUsed = self.scanFile()
    roi = RoiOperation().getRoi(rgbImage)
    roiBoundingBoxes = RoiOperation().getRoiBoundingBoxes(roi, rgbImage)
    positivePatchIndex = ExtractPatches().extractNegativePatchesFromNormal(wsiSlide, roi, levelUsed, roiBoundingBoxes,
                                                                         patchNormalNegativeSaveDir, self.positivePatchIndex)
    return positivePatchIndex

  def scanFile(self):
    wsiSlide = openslide.OpenSlide(self.slideDir)
    levelUsed = wsiSlide.level_count - 2
    rgbImage = np.array(wsiSlide.read_region((0, 0), levelUsed, wsiSlide.level_dimensions[levelUsed]))
    return wsiSlide, rgbImage, levelUsed

class TumorPreprocessing(object):
  def __init__(self, slideDir, xmlDir, positivePatchIndex, negativePatchIndex):
    self.slideDir = slideDir
    self.xmlDir = xmlDir
    self.positivePatchIndex = positivePatchIndex
    self.negativePatchIndex = negativePatchIndex
    pass

  def tumorPreprocessing(self):
    wsiSlide, rgbImage, roiLevel = self.scanFile()
    roi = RoiOperation().getRoi(rgbImage)
    roiBoundingBoxes = RoiOperation().getRoiBoundingBoxes(roi, rgbImage)

    # Use the higher level to get the accurate tumor
    tumorLevel = 3
    tumorLevelContours = TumorMaskOperation().getTumorContours(xmlDir, tumorLevel)
    tumorMask = TumorMaskOperation().getTumorMask(wsiSlide, tumorLevelContours, tumorLevel)
    # Need to be resize the image to the ROI level
    resizeFactor = float(1.0 / pow(2, roiLevel - tumorLevel))
    tumorMask = cv2.resize(np.array(tumorMask), (0, 0), fx = resizeFactor, fy = resizeFactor)
    # Bounding box is also in ROI level
    roiLevelContours = TumorMaskOperation().getTumorContours(xmlDir, roiLevel)
    tumorBoundingBoxes = TumorMaskOperation().getTumorBoundingBoxes(roiLevelContours)

    positivePatchIndex = ExtractPatches().extractPositivePatchesFromTumor(wsiSlide, tumorMask, roiLevel, tumorBoundingBoxes,
                                                                        patchTumorPositiveSaveDir, self.positivePatchIndex)

    negativePatchIndex = ExtractPatches().extractNegativePatchesFromTumor(wsiSlide, tumorMask, roi, roiLevel, roiBoundingBoxes,
                                                                        patchTumorNegativeSaveDir, self.negativePatchIndex)

    return positivePatchIndex, negativePatchIndex

  def scanFile(self):
    wsiSlide = openslide.OpenSlide(self.slideDir)
    roiLevel = wsiSlide.level_count - 2
    rgbImage = np.array(wsiSlide.read_region((0, 0), roiLevel, wsiSlide.level_dimensions[roiLevel]))
    return wsiSlide, rgbImage, roiLevel

tumor = TumorPreprocessing(slideDir, xmlDir, 0, 0)
tumor.tumorPreprocessing()
