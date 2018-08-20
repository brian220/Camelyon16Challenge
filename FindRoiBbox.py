from PIL import Image
import cv2
import numpy as np

class FindRoiBbox(object):
  def __init__(self):
    pass

  def findRoiBbox(self, rgbImage):
    openImage = self.imagePreprocessing(rgbImage)
    boundingBoxes = self.getBoundingBoxes(openImage, rgbImage)
    return openImage, boundingBoxes

  def imagePreprocessing(self, rgbImage):
    hsvImage = self.rgbToHsv(rgbImage)
    maskImage = self.binaryMaskGeneration(hsvImage)
    closeImage = self.closing(maskImage)
    openImage = self.opening(closeImage)
    return openImage

  def rgbToHsv(self, rgbImage):
    hsvImage = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2HSV)
    #self.saveImage(hsvImage, "hsv")
    return hsvImage

  # set the color in the range lowerRed, upperRed 255, else 0
  def binaryMaskGeneration(self, hsvImage):
    lowerRed = np.array([20, 20, 20])
    upperRed = np.array([200, 200, 200])
    maskImage = cv2.inRange(hsvImage, lowerRed, upperRed)
    #self.saveImage(maskImage, "mask")
    return maskImage

  def closing(self, maskImage):
    closeKernel = np.ones((20, 20), dtype=np.uint8)
    closeImage = cv2.morphologyEx(np.array(maskImage), cv2.MORPH_CLOSE, closeKernel)
    return closeImage

  def opening(self, closeImage):
    openKernel = np.ones((5, 5), dtype=np.uint8)
    openImage = cv2.morphologyEx(np.array(closeImage), cv2.MORPH_OPEN, openKernel)
    #self.saveImage(openImage, "closeOpenImage")
    return openImage

  def getBoundingBoxes(self, openImage, rgbImage):
    _, contours, _ = cv2.findContours(openImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rgbContour = None
    rgbContour = rgbImage.copy()
    lineColor = (255, 0, 0)  # blue color code
    cv2.drawContours(rgbContour, contours, -1, lineColor, 2)
    #self.saveImage(rgbContour, "rgbContour")
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    return boundingBoxes

  def saveImage(self, save, name):
    img = Image.fromarray(save)
    img.save(name + ".png")
