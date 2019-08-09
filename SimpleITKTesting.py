import SimpleITK as sitk
import numpy as np
import math
import matplotlib.pyplot as pyplot
import cv2

OUTPUT_DIR = "Output"

def distanceOf(p1: tuple = None, p2: tuple = None):
	x1, y1 = p1
	x2, y2 = p2
	return round(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))

def getExtrema(filename):
	file = cv2.imread(filename, 0)
	ret, bin = cv2.threshold(file, 127, 255, cv2.THRESH_BINARY_INV)
	contours = cv2.findContours(bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = imutils.grab_contours(contours)
	c = max(contours, key = cv2.contourArea)

	l = tuple(c[c[:, :, 0].argmin()][0])
	r = tuple(c[c[:, :, 0].argmax()][0])
	t = tuple(c[c[:, :, 1].argmin()][0])
	b = tuple(c[c[:, :, 1].argmax()][0])

	print("Left: ", l, ", Right: ", r, ", Top: ", t, ", Bottom: ", b, sep = "")
	return (l, r, t, b)

class RecordCoords:

	X = Y = 0
	Coords = []
	Count = 1

	def drawCircle(e, x, y, flags, param):
		if e == cv2.EVENT_LBUTTONDOWN:
			print(x, ",", y)
			X = x
			Y = y
			Coords.append([x, y])
			c = int(Count / 2) + 1
			if Count % 2 == 0: print("\nOriginal image value (", c, "): ", sep = "")
			else: print("\nCorresponding image value (", c, "): ", sep = "")
			Count += 1
