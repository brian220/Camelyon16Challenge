from PIL import Image
import cv2
import numpy as np

class RoiOperation(object):
  def __init__(self):
    pass

  # In order to get rid of the background image, we should find ROI (Region Of Interest)
  # ROI is a binary image, the background is black(0), ROI part is white(255)
  def getRoi(self, rgbImage):
    hsvImage = ImageProcessing().rgbToHsv(rgbImage)
    maskImage = ImageProcessing().binaryMaskGeneration(hsvImage)
    closeImage = ImageProcessing().closing(maskImage)
    Roi = ImageProcessing().opening(closeImage)
    return Roi

  # get the bounding boxes of ROI
  def getRoiBoundingBoxes(self, roi, rgbImage):
    _, contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rgbContour = None
    rgbContour = rgbImage.copy()
    lineColor = (255, 0, 0)  # blue color code
    cv2.drawContours(rgbContour, contours, -1, lineColor, 2)
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    return boundingBoxes

class ImageProcessing(object):
  def __init__(self):
      pass

  def rgbToHsv(self, rgbImage):
    self.saveImage(rgbImage, "rgb")
    hsvImage = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2HSV)
    return hsvImage

  # set the color in the range lowerRed, upperRed 255, else 0
  def binaryMaskGeneration(self, hsvImage):
    lowerRed = np.array([20, 20, 70])
    upperRed = np.array([200, 200, 200])
    maskImage = cv2.inRange(hsvImage, lowerRed, upperRed)
    return maskImage

  def closing(self, maskImage):
    closeKernel = np.ones((20, 20), dtype=np.uint8)
    closeImage = cv2.morphologyEx(np.array(maskImage), cv2.MORPH_CLOSE, closeKernel)
    return closeImage

  def opening(self, closeImage):
    openKernel = np.ones((5, 5), dtype=np.uint8)
    roi = cv2.morphologyEx(np.array(closeImage), cv2.MORPH_OPEN, openKernel)
    #self.saveImage(Roi, "Roi")
    return roi

  def saveImage(self, save, name):
    img = Image.fromarray(save)
    img.save(name + ".png")
