import numpy as np
import cv2

def get_transform(req_cnt, img):
	pts = req_cnt.reshape(4, 2)
	rect = np.zeros((4, 2), dtype="float32")

	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	widthA = np.sqrt((rect[0][0] - rect[1][0])**2 + (rect[0][1] - rect[1][1])**2)
	widthB = np.sqrt((rect[2][0] - rect[3][0])**2 + (rect[2][1] - rect[3][1])**2)
	heightA = np.sqrt((rect[0][0] - rect[3][0])**2 + (rect[0][1] - rect[3][1])**2)
	heightB = np.sqrt((rect[1][0] - rect[2][0])**2 + (rect[1][1] - rect[2][1])**2)

	width = max(int(widthA), int(widthB))
	height = max(int(heightA), int(heightB))

	dest = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
	M = cv2.getPerspectiveTransform(rect, dest)
	warp = cv2.warpPerspective(img, M, (width, height))
	return warp.copy()