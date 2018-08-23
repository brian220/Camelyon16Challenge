import sys
sys.path.append("../RoiOperation")
from RoiOperation import ImageProcessing
import cv2
import numpy as np
import xml.etree.cElementTree as ET
from openslide import OpenSlide
from PIL import Image

class TumorMaskOperation(object):
  def __init__(self):
    pass

  # get Contours from xml annotation files
  def getTumorContours(self, fileName, levelUsed):
    downsample = pow(2, levelUsed)
    tree = ET.ElementTree(file = fileName)
    contours = []
    for elem in tree.iter("Annotation"):
      if(elem.get("Type") != "None"):
        contour = self.getContour(elem, downsample)
        contours.append(contour)
    return contours

  def getContour(self, elem, downsample):
    contour = []
    for coorElem in elem.iter("Coordinate"):
      X = coorElem.get("X")
      Y = coorElem.get("Y")
      x = int(float(X) / downsample)
      y = int(float(Y) / downsample)
      contour.append([x, y])
    contour = np.array(contour , dtype=np.int32)
    return contour

  # get the mask Image from the contours
  def getTumorMask(self, slide, contours, levelUsed):
    slideWidth = slide.level_dimensions[levelUsed][0]
    slideHeight = slide.level_dimensions[levelUsed][1]
    maskShape = (slideHeight, slideWidth)
    tumorMask = self.makeMask(maskShape, contours)
    self.saveImage(tumorMask, "tumormask")
    return tumorMask

  def makeMask(self, maskShape, contours):
    tumorMask = np.zeros(maskShape[:2])
    tumorMask = tumorMask.astype(np.uint8)
    lineColor = (255, 255, 255)
    cv2.drawContours(tumorMask, contours, -1, lineColor, -1)
    return tumorMask

  # Need to resize the image to the ROI level
  def resizeTumorMask(self, tumorMask, roiLevel, tumorLevel):
    resizeFactor = float(1.0 / pow(2, roiLevel - tumorLevel))
    tumorMask = cv2.resize(np.array(tumorMask), (0, 0), fx = resizeFactor, fy = resizeFactor)
    return tumorMask

  # get the bounding boxes from the contours
  def getTumorBoundingBoxes(self, contours):
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    return boundingBoxes

  def saveImage(self, save, name):
    img = Image.fromarray(save)
    img.save(name + ".png")
