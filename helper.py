def canPlace(board, x, y, num):
	for i in board:
		if i[y] == num:
			return False
	for i in board[x]:
		if i == num:
			return False
	box_rst = (x//3)*3
	box_cst = (y//3)*3
	for i in range(box_rst, box_rst+3):
		for j in range(box_cst, box_cst+3):
			if board[i][j] == num:
				return False
	return True

def solve_sudoku(board, x, y):
	if x==9:
		return True
	if y==9:
		return solve_sudoku(board, x+1, 0)
	if board[x][y]!=0:
		return solve_sudoku(board, x, y+1)
	for num in range(1, 10):
		if canPlace(board, x, y, num):
			board[x][y] = num
			wasSolved = solve_sudoku(board, x, y+1)
			if wasSolved == True:
				return True
			board[x][y] = 0
	return False
