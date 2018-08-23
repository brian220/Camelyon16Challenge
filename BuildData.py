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
    wsiPaths = self.getFilePaths(utils.TUMOR_TRAINNING_SLIDE_DIR, '*.tif')
    xmlPaths = self.getFilePaths(utils.XML_DIR, '*.xml')
    for wsiPath, xmlPath in zip(wsiPaths, xmlPaths):
      self.printTumorSlideMessage(wsiPath, xmlPath)

      if self.fileMatchError(wsiPath, xmlPath):
        printMatchError()
        break
      else:
        self.getTumorData(wsiPath, xmlPath)

  def buildDataFromNormal(self):
    wsiPaths = self.getFilePaths(utils.NORMAL_TRAINNING_SLIDE_DIR, '*.tif')
    for wsiPath in wsiPaths:
      self.printNormalSlideMessage(wsiPath)
      self.getNormalData(wsiPath)

  def getFilePaths(self, fileDir, fileKind):
    paths = glob.glob(os.path.join(fileDir, fileKind))
    paths.sort()
    return paths

  def printTumorSlideMessage(self, wsiPath, xmlPath):
    print ("Current Process File Name:(Tumor Slide)")
    print ("WSI: " + wsiPath)
    print ("Annotation: " +  xmlPath)

  def printNormalSlideMessage(self, wsiPath):
    print ("Current Process File Name:(Normal Slide)")
    print ("WSI: " + wsiPath)

  def printMatchError(self):
    print("ERROR:Filename mismatch!.")

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

  def getNormalData(self, wsiPath):
    normal = NormalPreprocessing(wsiPath, self.positivePatchIndex)
    self.positivePatchIndex = normal.normalPreprocessing()

# Start the preprocessing
btd = BuildTrainningData()
btd.buildTrainningData()
