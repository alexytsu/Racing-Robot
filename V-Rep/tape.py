""" custom vision modules for tape detection """

import cv2
import numpy as np


def cannyFilter(frame):
    """Produces a dilated mask of edges in a frame """
    edges = cv2.Canny(frame, 50, 250)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    return edges


def colourFilter(frame):
    """Filters an image based for high saturation values """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowerSat = (0, 200, 0)
    higherSat = (255, 255, 255)
    newFrame = cv2.inRange(frame, lowerSat, higherSat)
    colorMask = cv2.cvtColor(newFrame, cv2.COLOR_GRAY2BGR)
    blueLower = (0, 200, 0)
    blueHigher = (60, 255, 255)
    yellowLower = (60, 200, 0)
    yellowHigher = (120, 255, 255)
    blueMask = cv2.inRange(frame, blueLower, blueHigher)
    yellowMask = cv2.inRange(frame, yellowLower, yellowHigher)
    return colorMask & frame, newFrame, yellowMask, blueMask


def applyROI(frame):
    """Filters out the part of the image that shows the car itself """
    frame[200:, 68:608] = 0
    return frame
