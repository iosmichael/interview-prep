class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return False
        row, col = len(matrix), len(matrix[0])
        c = row//2, col//2
        if row == 1 and col == 1:
            return target == matrix[0][0]
        if row == 2 or col == 2:
            c = (0 if row == 2 else row//2, 0 if row == 2 else row //2)
        m1, m4, m2, m3 = self.partition(matrix, c)
        if target < matrix[c[0]][c[1]]:
            print(m1)
            return self.searchMatrix(m1, target) 
        elif target > matrix[c[0]][c[1]]:
            return self.searchMatrix(m4, target) or self.searchMatrix(m2, target) or self.searchMatrix(m3, target)
        else:
            return True
    '''
    m11, m22, m12, m21
    '''
    def partition(self, m, c):
        c_x, c_y = c
        m1 = [r[:c_y+1] for r in m[:c_x+1]]
        m4 = [r[c_y:] for r in m[c_x:]]
        m2 = [r[c_y:] for r in m[:c_x+1]]
        m3 = [r[:c_y+1] for r in m[c_x:]]
        if m1 is None:
            m1 = [[]]
        if m4 is None:
            m4 = [[]]
        return m1, m4, m2, m3

def main():
    data = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16, 22],[10,13, 14, 17, 24],[18, 21, 23, 26, 30]]
    solu = Solution()
    print(solu.searchMatrix(data, 15))


if __name__ == '__main__':
    main()