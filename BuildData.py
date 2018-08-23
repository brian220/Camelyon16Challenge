"""
ICIAR2018 - Grand Challenge on Breast Cancer Histology Images
https://iciar2018-challenge.grand-challenge.org/home/
"""
import sys
sys.path.append("../Preprocessing")
sys.path.append("../Utils")
from Preprocessing import NormalPreprocessing, TumorPreprocessing
import Utils as utils
import glob
import os

class BuildTrainningData(object):
  def __init__(self):
    self.positivePatchIndex = 0
    self.negativePatchIndex = 0
    pass
  def buildTrainningData(self):
    self.buildDataFromTumor()
    self.buildDataFromNormal()

  def buildDataFromTumor(self):
    wsiPaths = glob.glob(os.path.join(utils.TUMOR_TRAINNING_SLIDE_DIR, '*.tif'))
    wsiPaths.sort()
    xmlPaths = glob.glob(os.path.join(utils.XML_DIR, '*.xml'))
    xmlPaths.sort()
    for wsiPath, xmlPath in zip(wsiPaths, xmlPaths):
      print ("Current Process File Name:(Tumor Slide)")
      print ("WSI: " + wsiPath)
      print ("Annotation: " +  xmlPath)

      if self.fileMatchError(wsiPath, xmlPath):
          print("ERROR:Filename mismatch!.")
          break
      else:
          self.getTumorData(wsiPath, xmlPath)

  def fileMatchError(self, wsiPath, xmlPath):
    matchError = False
    wsiLen = len(wsiPath)
    xmlLen = len(xmlPath)
    wsiName = wsiPath[wsiLen - 13: wsiLen - 4]
    xmlName = xmlPath[xmlLen - 13: xmlLen - 4]
    if wsiName != xmlName:
      matchError = True
    return matchError

  def getTumorData(self, wsiPath, xmlPath):
    tumor = TumorPreprocessing(wsiPath, xmlPath, self.positivePatchIndex, self.negativePatchIndex)
    self.positivePatchIndex, self.negativePatchIndex = tumor.tumorPreprocessing()

  def buildDataFromNormal(self):
    wsiPaths = glob.glob(os.path.join(utils.NORMAL_TRAINNING_SLIDE_DIR , '*.tif'))
    for wsiPath in wsiPaths:
      print ("Current Process File Name:(Normal Slide)")
      print ("WSI: " + wsiPath)
      self.getNormalData(wsiPath)

  def getNormalData(self, wsiPath):
    tumor = NormalPreprocessing(wsiPath, self.positivePatchIndex)
    self.positivePatchIndex = tumor.normalPreprocessing()

btd = BuildTrainningData()
btd.buildTrainningData()
