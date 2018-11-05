'''
McKinsey Software Engineering Intern Problem:

given a coordinate of water wall, calculate the maximum water capacity. Use O(n) runtime

(0, 9), (1, 5), (4, 8), (5, 2), (10, 3) => water
-
-***-
-***-
-***-
-----
-----*****-
-----------
-----------
-----------
'''

def max_water_capacity(test):
	meta_data = {}
	for i in range(len(test)):
		if i == 0:
			meta_data['max_x'] = test[0][0]
			meta_data['max_h'] = test[0][1]
			meta_data['water_cap'] = 0
			meta_data['filler_cap'] = 0
			continue
		x, h = test[i]
		if h < meta_data['max_h'] and h < test[i-1][1]:
			meta_data['filler_cap'] += h * (x - test[i-1][0])
		if h < meta_data['max_h'] and h > test[i-1][1]:
			meta_data['filler_cap'] += h * (x - test[i-1][0])
			meta_data['water_cap'] += (h - test[i-1][1]) * (x - test[i-1][0])
		if h >= meta_data['max_h']:
			meta_data['max_x'] = test[i][0]
			meta_data['max_h'] = test[i][1]
			meta_data['water_cap'] += meta_data['max_h'] * (x - meta_data['max_x'])
			meta_data['filler_cap'] = 0
	return meta_data['water_cap']

def main():
	test = [(0, 9), (1, 5), (4, 8), (5, 2), (10, 3)]
	print("Maximum Water Capacity is: {}".format(max_water_capacity(test)))

if __name__ == '__main__':
	main()