import cv2
import numpy as np
from os import listdir
import xml.etree.cElementTree as ET
from openslide import OpenSlide
from PIL import Image
filePathXml = "patient_004_node_4.xml"
slideDir = r'C:\Users\nctu\Desktop\Caymelon16Challenge\slide\patient_004_node_4.tif'

class TumorOperation(object):
  def __init__(self):
    pass

  def getContoursFromXml(self, fileName, levelUsed):
    downsample = pow(2, levelUsed)
    tree = ET.ElementTree(file = fileName)
    contours = []
    for elem in tree.iter("Annotation"):
      if(elem.get("Type") != "None"):
        contour = self.getContour(elem, downsample)
        contours.append(contour)
    print(contours)
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

  def getTumorMask(self, slide, contours, levelUsed):
    slideWidth = slide.level_dimensions[levelUsed][0]
    slideHeight = slide.level_dimensions[levelUsed][1]
    maskShape = (slideHeight, slideWidth)
    tumorMask = self.makeMask(maskShape, contours)
    self.saveImage(tumorMask, "mask")
    return tumorMask

  def makeMask(self, maskShape, contours):
    tumorMask = np.zeros(maskShape[:2])
    tumorMask = tumorMask.astype(np.uint8)
    lineColor = (255, 255, 255)
    cv2.drawContours(tumorMask, contours, -1, lineColor, -1)
    return tumorMask

  def saveImage(self, save, name):
    img = Image.fromarray(save)
    img.save(name + ".png")
