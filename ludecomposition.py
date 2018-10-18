import numpy as np

'''
this program is to reimplement LU decomposition for solving matrix A
- include forward substitution
- back substitution
- LU precomputing matrix factorization
'''

def forward_substitution(A, b):
	U, y = [row[:] for row in A], b[:]
	row, col = len(U), len(U[0])
	#pivot each row as the starting point for forward substitution
	for p in range(0, row):
		s = 1 / U[p][p]
		#scale the row to turn p^th element in pivot row into 1
		y[p] = s * y[p]
		for c in range(p, col):
			U[p][c] = s * U[p][c]
			#finish scaling the first row
		for r in range(p + 1, row):
			s = -U[r][p]
			y[r] = y[r] + s * y[p]
			for c in range(p, col):
				U[r][c] = U[r][c] + s * U[p][c]
	return U, y

def backward_substitution(U, y):
	'''
	Solve upper triangular systems Ux = y for x
	'''
	x = y[:]
	for p in range(0, len(y), -1):
		#iterate backward over pivots
		for r in range(0, p):
			#eliminate values above u[p][p]
			x[r] = x[r] - U[r][p]*x[p]/U[p][p]
	return x

'''
criterion for LU decomposition:
- top-left submatrix kxk A_11 is nonsingular

LU decomposition should be unique for each factorization process
'''

def LU_factorization_compact(A):
	'''
	A is a nxn matrix
	this method factors A to A = LU in compact format
	'''
	# if len(A) < 1:
	# 	raise Exception
	A_ = [row[:] for row in A]
	row, col = len(A_), len(A_[0])
	L, U = [],[]
	for i in range(row):
		L.append([0] * col)
		U.append([0] * col)
		if i < col:
			L[i][i] = 1

	for i, e in enumerate(A_[0]):
		U[0][i] = e

	for p in range(0, col): #choose pivots like in forward substitution
	#after this iteration, the p^th element in the pivot column becomes 1
		for r in range(p+1, row):
			s = -A_[r][p]/A_[p][p]
			L[r][p] = -s
			A_[r][p] = -s
			for c in range(p+1, col):
				print("update pivot {} row {} col {}".format(p, r, c))
				if r <= c:
					#U matrix
					U[r][c] = A_[r][c] + s*A_[p][c]
				A_[r][c] = A_[r][c] + s*A_[p][c]
	#A_ is the compact version of factorization
	return A_, L, U

'''
mat is a 2D matrix
'''
def print_matrix(mat, name):
	for i, row in enumerate(mat):
		if i == 0:
			print("{}: [{}".format(name, row))
		elif i == len(mat) - 1:
			print("{}]".format(row))
		else:
			print("{}".format(row))

def main():
	A = [[2, 3, 4],[2, 5, 6],[3, 3, 2]]
	x = [2, 3, 10]
	lu_compact, L, U = LU_factorization_compact(A)
	print_matrix(A, "A")
	print("truth: {}".format(np.linalg.solve(A, x)))
	print("After LU decomposition (compact)\n")
	print_matrix(lu_compact, "LU Compact")
	U_tr, y = forward_substitution(A, x)
	x_vec = backward_substitution(U_tr, y)
	print("Gaussian result: {}".format(x_vec))
	U_tr, y = forward_substitution(L, x)
	x_vec = backward_substitution(U, y)
	print("LU decomposition result: {}".format(x_vec))


if __name__ == '__main__':
	main()