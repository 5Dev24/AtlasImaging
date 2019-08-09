import skimage as ski
from skimage.io import imread
from skimage.morphology import erosion, disk
from skimage import transform
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import numpy as np
import scipy.misc as msc
import math

FOLDER = "/home/anon/Pictures/"

ATLAS = {
	"TOP": [1500, 260],
	"LEFT": [60, 1005],
	"BOTTOM_RIDGE_1": [995, 1750],
	"BOTTOM_RIDGE_2": [2000, 1745],
	"RIGHT": [2945, 1025]
}

def trans(img: object = None, xaxis: int = 0, yaxis: int = 0):
	return transform.warp(img, transform.SimilarityTransform(translation = (-1 * xaxis, yaxis)))

def rightShift(img: object = None, amount: int = 0): return trans(img, amount, 0)

def leftShift(img: object = None, amount: int = 0): return rightShift(img, 0, -1 * amount, 0)

def upShift(img: object = None, amount: int = 0): return trans(img, 0, amount)

def downShift(img: object = None, amount: int = 0): return upShift(img, -1 * amount)

def scaleBy(img: object = None, amount: tuple = None):
	h, w = img.shape
	tf = transform.AffineTransform(scale = amount)
	img = transform.warp(img, tf)
	h2, w2 = (int(h * (1 / amount[1])), int(w * (1 / amount[0])))
	zerov = generateZerosV(h)
	zeroh = generateZerosH(w)
	toadd = [(w - w2) / 2, (h - h2) / 2]
	while toadd[0] > 0:
		img = np.hstack((zerov, img))
		toadd[0] -= 1
	img = np.hsplit(img, np.array([w, h]))[0]
	while toadd[1] > 0:
		img = np.vstack((zeroh, img))
		toadd[1] -= 1
	img = np.vsplit(img, np.array([h, w]))[0]
	return img

def rotateBy(img: object = None, amount: int = 0): return transform.rotate(img, -1 * amount)

def generateZerosV(length: int = 0):
	lst = []
	for i in range(length): lst.append([0])
	return lst

def generateZerosH(length: int = 0):
	lst = []
	for i in range(length): lst.append(0)
	return [lst]

def cutImageAt(img: object = None, y: int = 0):
	return (
		np.vstack((
			img[:y],
			np.zeros_like(img[y:img.shape[0]])
		)),
		np.vstack((
			np.zeros_like(img[:y]),
			img[y:img.shape[0]]
		))
	)

def mergeAt(img1: object = None, img2: object = None, y: int = 0):
	print("egg")
	return msc.toimage(np.vstack((
		img1[:y],
		img2[y:]
	)))

def onClick(e: object = None):
	x, y = round(e.xdata, 2), round(e.ydata, 2)
	print("X: %.2f, Y: %.2f" % (x, y))

def main():
	base = imread(FOLDER + "base.jpg")
	atlas = imread(FOLDER + "atlas.jpg")
	atlas = erosion(atlas, disk(1))

	atlasTop, atlasBottom = cutImageAt(atlas, 1360)
	atlasBottom = downShift(atlasBottom, 50)
	atlas = mergeAt(atlasTop, atlasBottom, 1360)
	"".title()

	fig = plt.figure(1)

	plt.imshow(base, alpha = 1, cmap = "gray")
	plt.imshow(atlas, alpha = .8, cmap = "gray")

	plt.axis("off")
	plt.subplots_adjust(left = 0, right = 1, top = 1, bottom = 0)
	fig.set_size_inches(base.shape[0] / 80, base.shape[1] / 80)
	fig.canvas.mpl_connect("button_press_event", onClick)

	plt.show()

if __name__ == "__main__": main()
