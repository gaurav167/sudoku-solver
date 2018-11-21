import numpy as np
import cv2

from skimage.feature import hog
from sklearn.externals import joblib

from warpTransform import get_transform

def getSudoku(filename):
	## Get Transformed Sudoku Image

	# Read image -> GrayScale -> Threshold -> Detect Edges -> Transform view
	img = cv2.imread(filename)
	# gray = cv2.imread("sudoku001.jpg", 0)
	# cv2.imshow("img", img)
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	gray = cv2.GaussianBlur(gray, (5,5), 0)

	gaus = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 111, 12)
	# cv2.imshow("gaus", gaus)
	# cv2.moveWindow("gaus", 0, 0)

	kernel = np.ones((3,3), np.uint8)
	erd = cv2.erode(gaus, kernel, iterations = 1)
	# cv2.imshow("erd", erd)
	# cv2.moveWindow("erd", img.shape[1], 0)

	kernel = np.ones((5,5), np.uint8)
	eroded1 = cv2.erode(erd, kernel)
	dilated1 = cv2.dilate(erd, kernel)
	edges = cv2.absdiff(eroded1, dilated1)
	# cv2.imshow("edge", edges)

	_, ctr, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	ctr = sorted(ctr, key = cv2.contourArea, reverse=True)
	req_cnt = []
	for c in ctr:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		if len(approx) == 4:
			cv2.drawContours(gray, approx, -1, (0,255,0), 4)
			req_cnt = approx
			break

	sudoku_area = cv2.contourArea(req_cnt)

	warp = get_transform(req_cnt, img)

	# cv2.imshow("warp", warp)
	# cv2.moveWindow("warp", 0, img.shape[0])
	# cv2.imshow("gray", gray)
	# cv2.moveWindow("gray", 3*img.shape[1], 0)
	return warp, sudoku_area


def getDigitBoxes(warp, sudoku_area):
	## Get indivisual digit boxes

	hsv = cv2.cvtColor(warp, cv2.COLOR_RGB2HSV)
	black = cv2.inRange(hsv, (0,0,0), (180,255,100))
	# cv2.imshow("hsv", black)
	mask = np.ones((warp.shape[0], warp.shape[1]), np.uint8)*255
	im = cv2.bitwise_and(black, mask)
	# cv2.imshow("im", im)
	sudoku = im.copy()

	kernel = np.ones((3,3), np.uint8)
	a1 = cv2.erode(sudoku, kernel)
	a2 = cv2.dilate(sudoku, kernel)
	sudoku = cv2.absdiff(a1,a2)
	_, cont, _ = cv2.findContours(sudoku, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cont = sorted(cont, key=cv2.contourArea)
	boxes = []
	req_area = sudoku_area / 81
	widths=[]
	heights=[]
	for c in cont:
		if len(boxes) == 81:
			break
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.05*peri, True)
		if len(approx) == 4 and 10*cv2.contourArea(approx) > 6*req_area and 10*cv2.contourArea(approx) < 20*req_area :
			cs = approx.reshape(4,2)
			s = cs.sum(axis = 1)
			diff = np.diff(cs, axis=1)
			tl = cs[np.argmin(s)]
			br = cs[np.argmax(s)]
			tr = cs[np.argmin(diff)]
			bl = cs[np.argmax(diff)]
			heights.append((bl[1]-tl[1] + br[1]-tr[1])/2)
			widths.append((tr[0]-tl[0] + br[0]-bl[0])/2)
			boxes.append((tl.copy(), approx))
	# cv2.imshow("sudoku", sudoku)
	# cv2.moveWindow("sudoku", img.shape[1], img.shape[0])

	avg_height = sum(heights)//len(heights)
	avg_width = sum(widths)//len(widths)

	for box in boxes:
		box[0][0] = box[0][0] + avg_height - (box[0][0]%avg_height)
		box[0][1] = box[0][1] + avg_width - (box[0][1]%avg_width)
		
	boxes = sorted(boxes, key=lambda tup:(tup[0][1], tup[0][0]))

	return boxes, avg_width, avg_height

def makePuzzle(warp, boxes, avg_width, avg_height):
	## Read the boxes to get digits
	clf, pp = joblib.load("digits_cls_github.pkl")
	puzzle = [[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]
	for cc in boxes:
		num_warp = get_transform(cc[1], warp)
		num_warp = cv2.cvtColor(num_warp, cv2.COLOR_BGR2GRAY)
		num_warp = cv2.GaussianBlur(num_warp, (2*(num_warp.shape[0]//20)+1,2*(num_warp.shape[1]//20)+1), 0)
		num_warp = cv2.adaptiveThreshold(num_warp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 111, 12)
		_, cn, _ = cv2.findContours(num_warp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		roi = []
		cn = sorted(cn, key=cv2.contourArea)
		for c in cn:
			if cv2.contourArea(c) > num_warp.shape[0]*num_warp.shape[1]*0.05 and cv2.contourArea(c) < num_warp.shape[0]*num_warp.shape[1]*0.8:
				[x,y,w,h] = cv2.boundingRect(c)
				if h>num_warp.shape[1]*0.3:
					# cv2.rectangle(num_warp, (x,y), (x+w, y+h), (0,255,0),2)
					roi = num_warp[y:y+h, x:x+w]
					roi = cv2.resize(roi, (28,28))
					roi = np.array(roi)
					hog_ft = hog(roi,block_norm="L1", orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
					hog_ft = pp.transform(np.array([hog_ft], 'float64'))
					predicted = clf.predict(hog_ft)
					puzzle[max(0,min(int(cc[0][1]/avg_width)-1,8))][max(0,min(8,int(cc[0][0]/avg_height)-1))] = int(predicted[0])
					break
	return puzzle
