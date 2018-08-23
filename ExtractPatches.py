import sys
sys.path.append("../Utils")
import Utils as utils
import numpy as np

class ExtractPatches(object):
  def __init__(self):
      pass

  def extractPositivePatchesFromTumor(self, wsiSlide, tumorMask, levelUsed, tumorBoundingBoxes, patchIndex):
    magFactor = pow(2, levelUsed)
    for tumorBoundingBox in tumorBoundingBoxes:
      X, Y = self.getRandomPointsInBoundingBox(tumorBoundingBox, utils.NUM_POSITIVE_PATCHES_FROM_EACH_BBOX)
      for x, y in zip(X, Y):
        if int(tumorMask[y, x]) is utils.PIXEL_WHITE:
          self.savePatch(x, y, magFactor, wsiSlide,  utils.PATCH_POSITIVE_SAVE_DIR, "Positive", patchIndex)
          patchIndex += 1
    return patchIndex

  def extractNegativePatchesFromTumor(self, wsiSlide, tumorMask, roi, levelUsed, roiBoundingBoxes, patchIndex):
    magFactor = pow(2, levelUsed)
    for roiBoundingBox in roiBoundingBoxes:
      X, Y = self.getRandomPointsInBoundingBox(roiBoundingBox, utils.NUM_NEGATIVE_PATCHES_FROM_EACH_BBOX)
      for x, y in zip(X, Y):
         if int(tumorMask[y, x]) is not utils.PIXEL_WHITE and int(roi[y, x]) is not utils.PIXEL_BLACK:
           self.savePatch(x, y, magFactor, wsiSlide, utils.PATCH_NEGATIVE_SAVE_DIR, "Negative", patchIndex)
           patchIndex += 1
    return patchIndex

  def extractNegativePatchesFromNormal(self, wsiSlide, roi, levelUsed, roiBoundingBoxes, patchIndex):
    magFactor = pow(2, levelUsed)
    for roiBoundingBox in roiBoundingBoxes:
      X, Y = self.getRandomPointsInBoundingBox(roiBoundingBox, utils.NUM_NEGATIVE_PATCHES_FROM_EACH_BBOX)
      for x, y in zip(X, Y):
        if int(roi[y, x]) is not utils.PIXEL_BLACK:
          self.savePatch(x, y, magFactor, wsiSlide,  utils.PATCH_NEGATIVE_SAVE_DIR, "Negative", patchIndex)
          patchIndex += 1
    return patchIndex

  def getRandomPointsInBoundingBox(self, boundingBox, patchesNumber):
    xStart = int(boundingBox[0])
    yStart = int(boundingBox[1])
    xEnd = int(boundingBox[0]) + int(boundingBox[2])
    yEnd = int(boundingBox[1]) + int(boundingBox[3])
    X = np.random.random_integers(xStart, high = xEnd - 1, size = patchesNumber)
    Y = np.random.random_integers(yStart, high = yEnd - 1, size = patchesNumber)
    return X, Y

  def savePatch(self, x, y, magFactor, wsiSlide, patchSaveDir, patchPrefix, patchIndex):
    patch = wsiSlide.read_region((x * magFactor, y * magFactor), 0, (utils.PATCH_SIZE, utils.PATCH_SIZE))
    patch.save(str(patchSaveDir) + '\\' + patchPrefix + str(patchIndex) + '.png')
    patch.close()
