'''
Coding challenges: build a max heap tree in 10 mins
Michael Liu
'''
class Heap(object):
	def __init__(self, data):
		#starting index is 1
		self.data = [0]
		self.nums = 0
		for dat in data:
			self.insert(dat)

	def heapify(self, start):
		if start >= len(self.data):
			return
		left_child_val = right_child_val = float('-inf')
		if self.left_child(start) < len(self.data):
			left_child_val = self.data[self.left_child(start)]
		if self.right_child(start) < len(self.data):
			right_child_val = self.data[self.right_child(start)]
		if self.data[start] < max(left_child_val, right_child_val):	
			if left_child_val > right_child_val:
				self.data[start], self.data[self.left_child(start)] = left_child_val, self.data[start]
				self.heapify(self.left_child(start))
			else:
				self.data[start], self.data[self.right_child(start)] = right_child_val, self.data[start]
				self.heapify(self.right_child(start))

	def insert(self, i):
		self.data.append(i)
		if self.nums != 0:
			self.data[1], self.data[len(self.data) - 1] = self.data[len(self.data)-1], self.data[1]
		self.nums += 1
		self.heapify_recursive(1)

	def heapify_recursive(self, i):
		self.heapify(i)
		if self.right_child(i) < len(self.data):
			self.heapify_recursive(self.right_child(i))
		if self.left_child(i) < len(self.data):
			self.heapify_recursive(self.left_child(i))

	def print_heap(self, start):
		queue = [("", start)]
		while len(queue) != 0:
			prefix, node = queue.pop(0)
			print(prefix, self.data[node])
			if self.left_child(node) < len(self.data):
				queue.append((prefix + "-", self.left_child(node)))
			if self.right_child(node) < len(self.data):
				queue.append((prefix + "-", self.right_child(node)))

	def left_child(self, n):
		return 2 * n

	def right_child(self, n):
		return 2 * n + 1

	def get_max(self):
		max_num = self.data.pop(1)
		self.nums -= 1
		self.heapify_recursive(1)
		return max_num

def main():
	data = [1, 2, 5, 2, 3]
	heap = Heap(data)
	heap.print_heap(1)
	for i in range(heap.nums):
		print(heap.get_max())

if __name__ == '__main__':
	main()