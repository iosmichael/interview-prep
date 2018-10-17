'''
McKinsey Software Engineering Intern Problem 1:

There are two lakes in an area map. In order to connect the two lakes, a canal need to be digged 
between the two lakes. Diagnal canal digging doesn't count.

0 is lake
1 is land

find the shortest canal to dig between the two lakes

[[0 0 0 1 1 1]
 [0 1 1 1 0 0]
 [1 1 1 1 0 0]
 [1 0 0 0 0 0]]

shortest distance is 2: (1, 2) (1, 3)
'''
def find_shortest(area_map):
	'''
	m is row
	n is col
	'''
	m, n = len(area_map), len(area_map[0])
	start_i, start_j = 0, 0
	is_found = False
	for i in range(0, m):
		if is_found == True:
			break
		for j in range(0, n):
			if area_map[i][j] == 0:
				start_i, start_j = i, j
				is_found = True
				break

	print("starting point: {}, {}: {}".format(start_i, start_j, area_map[start_i][start_j]))
	'''
	run dijsktra algorithm
	with i+, i- and j-, j+
	'''
	visited = []
	d = []
	for i in range(m):
		c = []
		c_v = []
		for j in range(n):
			c.append(m * n + 1)
			c_v.append(False)
		d.append(c)
		visited.append(c_v)

	d[start_i][start_j] = 0
	p_queue = [(start_i, start_j, 0)]
	while len(p_queue) > 0:
		p_queue = sorted(p_queue, key=lambda x: x[2])
		i, j, e = p_queue.pop(0)
		# print_matrix(d)
		visited[i][j] = True
		nodes = children_nodes(i, j, m, n)
		for node in nodes:
			flex_node(d, area_map[node[0]][node[1]], i, j, node[0], node[1])
			if visited[node[0]][node[1]] == False:
				p_queue.append((node[0], node[1], area_map[node[0]][node[1]]))
	return min([d[i][j] for i in range(m) for j in range(n) if d[i][j] > 0 and area_map[i][j] != 1])

def children_nodes(i, j, m, n):
	nodes = []
	if i - 1 >= 0:
		nodes.append((i - 1, j))
	if i + 1 < m:
		nodes.append((i + 1, j))
	if j - 1 >= 0:
		nodes.append((i, j - 1))
	if j + 1 < n:
		nodes.append((i, j + 1))
	return nodes

def flex_node(d, e, i, j, x, y):
	val = d[i][j] + e
	if d[x][y] > val:
		d[x][y] = val

def print_matrix(d):
	for l in d:
		print(l)
	print('\n\n')

def main():
	area_map = [[0, 0, 0, 1, 1, 1], [0, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0], [1, 0, 0, 0, 0, 0]]
	s_n = find_shortest(area_map)
	print("shortest path map: {}".format(s_n))

if __name__ == '__main__':
	main()