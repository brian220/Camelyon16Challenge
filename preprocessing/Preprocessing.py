import sys
sys.path.append("../RoiOperation")
sys.path.append("../TumorMaskOperation")
sys.path.append("../ExtractPatches")
from RoiOperation import RoiOperation
from TumorMaskOperation import TumorMaskOperation
from ExtractPatches import ExtractPatches
import openslide
from openslide import open_slide # http://openslide.org/api/python/
import numpy as np

class NormalPreprocessing(object):
  def __init__(self, slideDir, positivePatchIndex):
    self.slideDir = slideDir
    self.positivePatchIndex = positivePatchIndex
    pass

  def normalPreprocessing(self):
    wsiSlide, rgbImage, levelUsed = self.scanFile()
    roi = RoiOperation().getRoi(rgbImage)
    roiBoundingBoxes = RoiOperation().getRoiBoundingBoxes(roi, rgbImage)
    positivePatchIndex = ExtractPatches().extractNegativePatchesFromNormal(wsiSlide, roi, levelUsed, roiBoundingBoxes, self.positivePatchIndex)
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

    tumorMask = self.getPreciseTumorMask(wsiSlide, roiLevel, 3)
    tumorBoundingBoxes = self.getRoiLevelTumorBoundingBoxes(roiLevel)

    positivePatchIndex = ExtractPatches().extractPositivePatchesFromTumor(wsiSlide, tumorMask, roiLevel, tumorBoundingBoxes, self.positivePatchIndex)
    negativePatchIndex = ExtractPatches().extractNegativePatchesFromTumor(wsiSlide, tumorMask, roi, roiLevel, roiBoundingBoxes, self.negativePatchIndex)
    return positivePatchIndex, negativePatchIndex

  def scanFile(self):
    wsiSlide = openslide.OpenSlide(self.slideDir)
    roiLevel = wsiSlide.level_count - 2
    rgbImage = np.array(wsiSlide.read_region((0, 0), roiLevel, wsiSlide.level_dimensions[roiLevel]))
    return wsiSlide, rgbImage, roiLevel

  # Use the higher level to get the tumor, which has more precise shape
  def getPreciseTumorMask(self, wsiSlide, roiLevel, preciseLevel):
     tumorLevelContours = TumorMaskOperation().getTumorContours(self.xmlDir, preciseLevel)
     tumorMask = TumorMaskOperation().getTumorMask(wsiSlide, tumorLevelContours, preciseLevel)
     tumorMask = TumorMaskOperation().resizeTumorMask(tumorMask, roiLevel, preciseLevel)
     return tumorMask

  # Bounding box is still in ROI level
  def getRoiLevelTumorBoundingBoxes(self, roiLevel):
    roiLevelContours = TumorMaskOperation().getTumorContours(self.xmlDir, roiLevel)
    tumorBoundingBoxes = TumorMaskOperation().getTumorBoundingBoxes(roiLevelContours)
    return tumorBoundingBoxes
