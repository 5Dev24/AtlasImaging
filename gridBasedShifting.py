import cv2
import imutils
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt

DEBUG = 2

class Chunk:

	def __init__(self, x: int = 0, y: int = 0, maxes: tuple = None, imageData: list = None):
		w, h = imageData.shape[:2][::-1]
		if x + w > maxes[0]: w -= x
		if y + h > maxes[1]: h -= y
		self.corners = [
			[x, y], # Top Left
			[x + w, y], # Top Right
			[x, y + h], # Bottom Left
			[x + w, y + h] # Bottom Right
		]
		self._initCorners = ((x, y), (x + w, y), (x, y + h), (x + w, y + h))
		self._imgData = imageData

class Image:

	def __init__(self, w: int = 0, h: int = 0, chunkW: int = 0,
				chunkH: int = 0, remainderW: int = 0, remainderH: int = 0,
				chunks: list = None):
		self._dimensions = (w, h)
		self._dimensionsWithoutRemainders = (w - remainderW, h - remainderH)
		self._chunkDimensions = (chunkW, chunkH)
		self._remainders = (remainderW, remainderH)
		self._chunks = chunks

	def draw(self, displayGrid: bool = False):
		emptyImage = np.zeros((self._dimensions[1], self._dimensions[0], 4), np.uint8)
		for chunkList in self._chunks:
			for chunk in chunkList:
				TOP = chunk.corners[0]
				BOTTOM = chunk.corners[3]
				w, h = BOTTOM[0] - TOP[0], BOTTOM[1] - TOP[1]
				emptyImage[TOP[1]:BOTTOM[1], TOP[0]:BOTTOM[0]] = chunk._imgData
				if displayGrid:
					emptyImage[TOP[1]:BOTTOM[1], TOP[0]:TOP[0] + 1] = np.full((h, 1, 4), 255)
					emptyImage[TOP[1]:BOTTOM[1], BOTTOM[0] - 1:BOTTOM[0]] = np.full((h, 1, 4), 255)
					emptyImage[TOP[1]:TOP[1] + 1, TOP[0]:BOTTOM[0]] = np.full((1, w, 4), 255)
					emptyImage[BOTTOM[1] - 1:BOTTOM[1], TOP[0]:BOTTOM[0]] = np.full((1, w, 4), 255)
		if DEBUG >= 2: print("Done rebuilding image")
		cv2.imshow("Output", emptyImage)

		# https://www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/

	def _transformImageTo(self, tl: int = 0, tr: int = 0, bl: int = 0,
							br: int = 0, currentImage: object = None):
		rect = np.zeros((4, 2), dtype = "int32")
		points = np.array([tl, tr, bl, br], dtype = "int32")
		s = points.sum(axis = 1)
		rect[0] = points[np.argmin(s)]
		rect[2] = points[np.argmax(s)]
		d = np.diff(points, axis = 1)
		rect[1] = points[np.argmin(d)]
		rect[3] = points[np.argmax(d)]
		tl, tr, br, bl = rect
		dimA = (np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2)),
			np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2)))
		dimB = (np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2)),
			np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2)))
		maxes = (max(int(dimA[0]), int(dimB[0])),
			max(int(dimA[1]), int(dimB[1])))
		d = np.array([[0, 0],
			[maxes[0] - 1, 0],
			[maxes[0] - 1, maxes[1] - 1],
			[0, maxes[1] - 1]],
			dtype = "int32")
		m = cv2.getPerspectiveTransform(rect, d)
		return cv2.warpPerspective(currentImage, m, maxes)

def divideImage(img: object = None):
	w, h = img.shape[:2][::-1]
	partWLength, partHLength = max(2, len(str(w)) - 2), max(2, len(str(h)) - 2)
	partW, partH = w, h
	while len(str(partW).split(".")[0]) > partWLength: partW /= 2
	while len(str(partH).split(".")[0]) > partHLength: partH /= 2
	splitCounts = (int(w // partW), int(h // partH))

	if DEBUG >= 2:
		print("Base            => W: ", w, ", H: ", h, sep = "")
		print("Split Counts    => W: ", splitCounts[0], ", H: ", splitCounts[1], sep = "")
		print("Pre Rounded     => W: ", partW, ", H: ", partH, sep = "")
	partW, partH = int(str(partW).split(".")[0]), int(str(partH).split(".")[0])

	if DEBUG >= 2: print("Post Rounded    => W: ", partW, ", H: ", partH, sep = "")
	remainderSplit = (w - partW * splitCounts[0], h - partH * splitCounts[1])

	if DEBUG >= 2: print("Remainder Split => W: ", remainderSplit[0], ", H: ", remainderSplit[1], sep = "")
	if DEBUG >= 1:
		print("Splitting image of initial size of (", w, "x", h, ")", sep = "")
		print("Into splits of (", partW, "x", partH, ")", sep = "")
		print("With a rounding error of (", remainderSplit[0], "x", remainderSplit[1], ")", sep = "")

	totalChunksToGenerate = splitCounts[0] * splitCounts[1] + (splitCounts[0] if remainderSplit[0] > 0 else 0) + (splitCounts[1] if remainderSplit[1] > 0 else 0) + (1 if remainderSplit[0] > 0 and remainderSplit[1] > 0 else 0)
	MAXES = (w, h)

	FullImage = []

	genFullChunk = lambda minX, maxX, minY, maxY: img[minY:maxY, minX:maxX]
	for TopLeftY in customRange(0, h - remainderSplit[1] - partH, partH):
		BottomRightY = TopLeftY + partH
		FullRowOfChunks = []
		for TopLeftX in customRange(0, w - remainderSplit[0] - partW, partW):
			BottomRightX = TopLeftX + partW
			FullRowOfChunks.append(Chunk(TopLeftX, TopLeftY, MAXES,
				genFullChunk(TopLeftX, BottomRightX, TopLeftY, BottomRightY)))
			updateCounter(FullRowOfChunks, FullImage, totalChunksToGenerate)
		if remainderSplit[0] > 0:
			FullRowOfChunks.append(Chunk(w - remainderSplit[0], TopLeftY, MAXES,
				genFullChunk(w - remainderSplit[0], w, TopLeftY, BottomRightY)))
			updateCounter(FullRowOfChunks, FullImage, totalChunksToGenerate)
		FullImage.append(FullRowOfChunks)
	if remainderSplit[1] > 1:
		FullRowOfChunks = []
		for TopLeftX in customRange(0, w - remainderSplit[0] - partW, partW):
			BottomRightX = TopLeftX + partW
			FullRowOfChunks.append(Chunk(TopLeftX, h - remainderSplit[1], MAXES,
				genFullChunk(TopLeftX, BottomRightX, h - remainderSplit[1], h)))
			updateCounter(FullRowOfChunks, FullImage, totalChunksToGenerate)
		if remainderSplit[0] > 0:
			FullRowOfChunks.append(Chunk(w - remainderSplit[0], h - remainderSplit[1], MAXES,
				genFullChunk(w - remainderSplit[0], w, h - remainderSplit[1], h)))
			updateCounter(FullRowOfChunks, FullImage, totalChunksToGenerate)
		FullImage.append(FullRowOfChunks)
	updateCounter([], FullImage, totalChunksToGenerate)
	if DEBUG >= 1: print("\nDone splitting image!")
	return Image(w, h, partW, partH, remainderSplit[0], remainderSplit[1], FullImage)

def updateCounter(currentFullRowOfChunks: list = None, fullImage: list = None, totalChunks: int = 0):
	if DEBUG >= 1: print(sum([len(chunkList) for chunkList in fullImage]) + len(currentFullRowOfChunks), "of", totalChunks, "Chunks Generated", end = "\r")

def customRange(start: int = 0, stop: int = 1, step = 1):
	vals = [i for i in range(start, stop, step)]
	if stop not in vals: vals.append(stop)
	return vals

def main():
	rawInput = cv2.imread("atlas.jpg")
	rawInput = np.dstack((rawInput, np.full(rawInput.shape[:-1], 125, dtype=np.uint8)))

	inputImgDivided = divideImage(rawInput)
	cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
	cv2.resizeWindow("Output", rawInput.shape[1], rawInput.shape[0])

	inputImgDivided.draw(True)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == "__main__": main()
