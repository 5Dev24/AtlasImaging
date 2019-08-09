def main():
	baseImg = cv2.resize(cv2.imread("base.jpg"), (600, 410))
	atlasImg = cv2.resize(cv2.imread("atlas.jpg"), (600, 410))
	kernels = (
		np.asarray([
			[0, 0, 1, 0, 0],
			[0, 1, 1, 1, 0],
			[1, 1, 1, 1, 1],
			[0, 1, 1, 1, 0],
			[0, 0, 1, 0, 0]
		], dtype = np.uint8),
		np.asarray([
			[0, 1, 1, 1, 0],
			[1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1],
			[0, 1, 1, 1, 0]
		], dtype = np.uint8)
	)
	baseImgBin = cv2.threshold(baseImg, 1, 255, cv2.THRESH_BINARY)[1]
	tmpImg = cv2.dilate(baseImgBin, kernels[0], iterations = 1)
	tmpImg = cv2.dilate(tmpImg, kernels[1], iterations = 2)
	tmpImg2 = cv2.erode(tmpImg, kernels[0], iterations = 1)
	newBaseImg = tmpImg - tmpImg2

	atlasGray = cv2.cvtColor(atlasImg, cv2.COLOR_BGR2GRAY)
	atlasFeatures = cv2.goodFeaturesToTrack(atlasGray, 3000, .01, 10)

	baseFeatures = np.array([])
	baseFeatures, pyrStati = cv2.calcOpticalFlowPyrLK(atlasImg, newBaseImg, atlasFeatures, baseFeatures, flags = 1)[:2]

	atlasFeaturesPruned = []
	baseFeaturesPruned = []
	for i, s in enumerate(pyrStati):
		if s == 1:
			baseFeaturesPruned.append(baseFeatures[i])
			atlasFeaturesPruned.append(atlasFeatures[i])

	baseFeaturesFinal = np.asarray(baseFeaturesPruned)
	atlasFeaturesFinal = np.asarray(atlasFeaturesPruned)

	trans, homStati = cv2.findHomography(baseFeaturesFinal, atlasFeaturesFinal, method = cv2.RANSAC, ransacReprojThreshold = 1)
	w, h = baseImg.shape[:2][::-1]
	modifiedImg = cv2.warpPerspective(atlasImg, trans, (w, h))
	modifiedImg = cv2.addWeighted(newBaseImg, .3, modifiedImg, 1, 1)
	outputImg = cv2.addWeighted(modifiedImg, 1, baseImg, .5, 1)

    cv2.imshow("Output", outputImg)
    cv2.waitKey(0)
	cv2.destroyAllWindows()
