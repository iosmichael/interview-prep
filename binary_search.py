'''
Coding challenge: build binary search from scratch within 10 min
Michael Liu
'''

def perform_binary_search_recursive(data, target):
	mid = int(len(data)/2)
	if len(data) < 1:
		return False
	if len(data) == 1 and data[0] != target:
		return False
	if target < data[mid]:
		return perform_binary_search_recursive(data[:mid], target)
	elif target > data[mid]:
		return perform_binary_search_recursive(data[mid:], target)
	else:
		return True

def perform_binary_search_linear(data, target):
	mid = int(len(data)/2)
	stack = [data]
	while (len(stack) > 0):
		sub_data = stack.pop(len(stack)-1)
		if len(sub_data) < 1:
			return False
		if len(sub_data) == 1:
			return sub_data[0] == target
		sub_mid = int(len(sub_data)/2)
		if sub_data[sub_mid] > target:
			stack.append((sub_data[:sub_mid]))
		elif sub_data[sub_mid] < target:
			stack.append((sub_data[sub_mid:]))
		else:
			return True
	return False

def main():
	target = 100
	data = [1, 2, 3, 5, 6, 8, 100]
	print(perform_binary_search_recursive(data, target))
	print(perform_binary_search_linear(data, target))

if __name__ == '__main__':
	main()