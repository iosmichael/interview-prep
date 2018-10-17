class Sorting(object):
	def __init__(self, data):
		self.data = data
		self.n = len(data)

	def merge_sort(self, data):
		pass

	def merge(self, arr1, arr2):
		pass

	def quick_sort(self, data, lo, hi):
		if lo < hi:
			p = self.partition(data, lo, hi)
			self.quick_sort(data, lo, p-1)
			self.partition(data, p+1, hi)
		return data

	def partition(self, data, lo, hi):
		pivot = data[hi]
		i = lo
		for j in range(lo, hi):
			if data[j] < pivot:
				data[i], data[j] = data[j], data[i]
				i += 1
		data[i], data[hi] = pivot, data[i]
		return i

def main():
	data = [2, 3, 6, 8, 1, 0, 3, 5]
	sorting = Sorting(data)
	print(sorting.quick_sort(data, 0, len(data)-1))



if __name__ == '__main__':
	main()