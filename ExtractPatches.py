import sys
sys.path.append("../Utils")
import Utils as utils
import numpy as np

class ExtractPatches(object):
  def __init__(self):
      pass

  def extractNegativePatchesFromNormal(self, wsiSlide, imageOpen, levelUsed,
                                       boundingBoxes, patchSaveDir, patchPrefix,
                                       patchIndex):
        magFactor = pow(2, levelUsed)

        print('No. of ROIs to extract patches from: %d' % len(boundingBoxes))

        for boundingBox in boundingBoxes:
            xStart = int(boundingBox[0])
            yStart = int(boundingBox[1])
            xEnd = int(boundingBox[0]) + int(boundingBox[2])
            yEnd = int(boundingBox[1]) + int(boundingBox[3])
            X = np.random.random_integers(xStart, high=xEnd, size=utils.NUM_NEGATIVE_PATCHES_FROM_EACH_BBOX)
            Y = np.random.random_integers(yStart, high=yEnd, size=utils.NUM_NEGATIVE_PATCHES_FROM_EACH_BBOX)

            for x, y in zip(X, Y):
                if int(imageOpen[y, x]) is not utils.PIXEL_BLACK:
                    patch = wsiSlide.read_region((x * magFactor, y * magFactor), 0,
                                                  (utils.PATCH_SIZE, utils.PATCH_SIZE))
                    patch.save(str(patchSaveDir) + '\\' + patchPrefix + str(patchIndex) + '.png')
                    patchIndex += 1
                    patch.close()
        return patchIndex
