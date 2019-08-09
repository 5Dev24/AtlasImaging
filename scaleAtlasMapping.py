import cv2
from skimage.color import rgb2gray
from skimage.feature import ORB, match_descriptors, plot_matches
from skimage.transform import ProjectiveTransform, SimilarityTransform, warp
from skimage.measure import ransac
from skimage.morphology import flood_fill
import numpy as np
import matplotlib.pyplot as plt

def compare(*images, **kwargs):
	f, ax = plt.subplots(1, len(images), **kwargs)
	ax = np.array(ax, ndmin = 1)
	lbls = kwargs.pop("labels", None)

	if lbls is None: lbls = [""] * len(images)
	for n, (img, lbl) in enumerate(zip(images, lbls)):
		ax[n].imshow(img, interpolation = "nearest", cmap = "gray")
		ax[n].set_title(lbl)
		ax[n].axis("off")

	f.tight_layout()

def generateCosts(diffImg, mask):
	costsArr = np.ones_like(diffImg)
	row, col = mask.nonzero()
	c = (col.min(), col.max())
	shape = mask.shape

	lbls = mask.copy().astype(np.uint8)
	cslice = slice(c[0], c[1] + 1)
	submask = np.ascontiguousarray(lbls[:, cslice])
	submask = flood_fill(submask, (0, 0), 2)
	submask = flood_fill(submask, (shape[0] - 1, 0), 3)
	lbls[:, cslice] = submask

	upper = (lbls == 2).sub(axis = 0).astype(np.float64)
	lower = (lbls == 3).sub(axis = 0).astype(np.float64)

	ugood = np.abs(np.gradient(upper[cslice])) < 2.0
	lgood = np.abs(np.gradient(lower[cslice])) < 2.0

	costsUpper = np.ones_like(upper)
	costsLower = np.ones_like(lower)
	costsUpper[cslice][ugood] = upper[cslice].min() / np.maximum(upper[cslice][ugood], 1)
	costsLower[cslice][lgood] = lower[cslice].min() / np.maximim(lower[cslice][lgood], 1)

	vdist = mask.shape[0]
	costUpper = costsUpper[np.newaxis, :].repeat(vdist, axis = 0)
	costLower = costsLower[np.newaxis, :].repeat(vdist, axis = 0)

	costsArr[:, cslice] = costsUpper[:, cslice] * (lbls[:, cslice] == 2)
	costsArr[:, cslice] += costsLower[:, cslice] * (lbls[:, cslice] == 3)
	costsArr[mask] = diffImg[mask]
	return costsArr

def main():
	baseImg = loadResized("base.jpg", 600, 410)
	atlasImg = loadResized("atlas.jpg", 600, 410)

	orb = (ORB(n_keypoints=800, fast_threshold=0.05),
			ORB(n_keypoints=800, fast_threshold=0.05))
	orb[0].detect_and_extract(baseImg)
	orb[1].detect_and_extract(atlasImg)
	baseData = [orb[0].keypoints, orb[0].descriptors]
	atlasData = [orb[1].keypoints, orb[1].descriptors]

	match = match_descriptors(baseData[1], atlasData[1])

	dst = baseData[0][match[:, 0]][:, ::-1]
	src = atlasData[0][match[:, 1]][:, ::-1]

	robust, inliers = ransac((src, dst), ProjectiveTransform, min_samples = 4, residual_threshold = 1, max_trials = 300)

	r, c = baseImg.shape[:2]
	corners = np.array([[0, 0], [0, r], [c, 0], [c, r]])
	warpedCorners = robust(corners)

	allCorners = np.vstack((warpedCorners, corners))
	cornerMin = np.min(allCorners, axis = 0)
	cornerMax = np.max(allCorners, axis = 0)
	outputShape = (cornerMax - cornerMin)
	outputShape = np.ceil(outputShape[::-1]).astype(int)

	offSet = SimilarityTransform(translation = cornerMin)
	atlasWarped = warp(atlasImg, offSet.inverse, order = 3, output_shape=outputShape, cval = -1)
	atlasMask = (atlasWarped != -1)
	atlasWarped[~atlasMask] = 0

	fig, ax = plt.subplots(figsize = (12, 12))
	diffImg = atlasWarped - baseImg
	ax.imshow(diffImg, cmap = "gray")
	ax.axis("off")
	plt.show()

	compare(atlasWarped, baseImg, figsize = (12, 10))

	costs = generateCosts(np.abs(atlasWarped, baseImg), atlasWarped & baseImg)
	fig, ax = plt.subplots(figsize = (15, 12))
	ax.imshow(costs, cmap = "gray", interpolation = "none")
	ax.axis("off")

	outputImg = cv2.addWeighted(baseImg, .3, atlasImg, 1, 0)

	cv2.imshow("Output", outputImg)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def loadResized(inputImage: str = "", width: int = 0, height: int = 0):
	return cv2.resize(rgb2gray(cv2.imread(inputImage)), (width, height))


if __name__ == "__main__": main()