import cv2
import numpy as np

def cannyFilter(frame):
    edges = cv2.Canny(frame, 50, 250)
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    return edges

def colourFilter(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_sat = (0, 200, 0)
    higher_sat = (255, 255, 255)
    newFrame = cv2.inRange(frame, lower_sat, higher_sat)
    colorMask = cv2.cvtColor(newFrame, cv2.COLOR_GRAY2BGR)
    blue_lower = (0, 200, 0)
    blue_higher = (60, 255, 255)
    yellow_lower = (60, 200, 0)
    yellow_higher = (120, 255, 255)
    blueMask = cv2.inRange(frame, blue_lower, blue_higher)
    yellowMask = cv2.inRange(frame, yellow_lower, yellow_higher)
    return colorMask & frame, newFrame, yellowMask, blueMask

def applyROI(frame):
    frame[200:, 68:608] = 0
    return frame