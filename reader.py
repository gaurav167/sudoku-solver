import numpy as np
import cv2

from functions import getSudoku, getDigitBoxes, makePuzzle
from helper import solve_sudoku

## Get Transformed Sudoku Image
warp, sudoku_area = getSudoku("sudoku.jpg")
## Get indivisual digit boxes
boxes, avg_width, avg_height = getDigitBoxes(warp, sudoku_area)
## Read the boxes to get digits and make game board
puzzle = makePuzzle(warp, boxes, avg_width, avg_height)

for i in puzzle:
	print(i)

## Solve the sudoku
solve = solve_sudoku(puzzle, 0, 0)

if solve == True:
	print("Solved")
	for i in puzzle:
		print(i)
else:
	print("Can't be solved.")