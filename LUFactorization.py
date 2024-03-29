import numpy as np

def swap_row(Matrix, row1, row2):
    """
    交换矩阵的两行
    :param Matrix: 要交换的矩阵
    :param row1: 行1
    :param row2: 行2
    :return: 交换后的矩阵
    """
    tmp = Matrix[row1].copy()
    Matrix[row1] = Matrix[row2]
    Matrix[row2] = tmp

    return Matrix

def getRowEchelonForm(Matrix):
    Matrix = Matrix.astype(np.float64)
    row_len, col_len = Matrix.shape
    L = np.zeros((row_len, row_len))
    U = Matrix.copy()
    index = [[x] for x in range(row_len)]

    for row in range(row_len - 1):
        # 如果该行全为0
        zero = np.sum(np.where(U[row, :], 0, 1))
        if zero == col_len:
            row_inverse = row_len - 1
            while row_inverse > row:
                # 该行零元素个数
                zero = np.sum(np.where(U[row_inverse, :], 0, 1))
                if zero != col_len:  # 该行不全为0，交换
                    swap_row(U, row_inverse, row)
                    swap_row(L, row_inverse, row)
                    swap_row(index, row_inverse, row)
                    #print(f"交换了 {row} 和 {row_inverse} 行")
                    break
                else:
                    row_inverse -= 1

            if row_inverse == row:  # 没有非0行了，退出
                # something to be fixed
                L = L + np.identity(row_len)
                return L, U, index

        # 找到第一个列号最小且不为0的行，并与当前行交换
        first_zero_pivot_index_set = {}
        for r in range(row, row_len):
            first_zero_pivot_index = row
            for col in range(row, col_len):
                if U[r][col] != 0:
                    break
                else:
                    first_zero_pivot_index += 1
            first_zero_pivot_index_set[r] = first_zero_pivot_index
        min_row = min(first_zero_pivot_index_set, key=first_zero_pivot_index_set.get)
        #print(first_zero_pivot_index_set)
        swap_row(U, row, min_row)
        swap_row(L, row, min_row)
        swap_row(index, row, min_row)
        #print(f"交换了 {row} 和 {min_row} 行")
        col = first_zero_pivot_index_set.get(min_row)
        #print(U)


        # 主元素为 augmented_matrix[row][col]
        for row2 in range(row + 1, row_len):
            coefficient = U[row2][col] / U[row][col]

            U[row2] = U[row2] - coefficient * U[row]
            #U[row2][col] = 0  # 把第一个元素置 0，不然可能会因为有效数问题导致消不掉？
            L[row2][row] = coefficient

    L = L + np.identity(row_len)
    #print("最后退出")
    return L, U, index


def isSolvable(matrix):
    row_len = matrix.shape[0]
    col_len = matrix.shape[1]
    # 统计零行
    zero_row = 0
    zero_aug_row = 0
    for row in range(row_len):
        # 统计零元素个数
        zero = np.sum(np.where(matrix[row][:-1], 0, 1))
        if zero == col_len - 1:
            if matrix[row][col_len - 1] == 0:
                zero_aug_row += 1
            zero_row += 1

    rank_A = row_len - zero_row
    rank_Ab = row_len - zero_aug_row
    print(f"Rank(A) = {rank_A}, Rank(A|b) = {rank_Ab}")
    if rank_A < rank_Ab:
        print("is not solvable")
    else:
        if rank_A < col_len - 1:
            print("infinte solutions")
        elif rank_A == col_len - 1:
            print("solvable")
        else:
            print("is not solvable")


if __name__ == "__main__":
    # 取消科学记数法显示
    np.set_printoptions(suppress=True)
    print("Test A mxm, r=m")
    Test_A = np.array([[2, 2, 2],
                  [4, 7, 7],
                  [6, 18, 22]])

    La, Ua, idxa = getRowEchelonForm(Test_A)
    print("L =\n", La)
    print("U =\n", Ua)
    print("A = LU = \n", La @ Ua)

    print("Test B mxn, m>n")
    Test_B = np.array([[2, 1, 1, 4],
                  [4, 2, 1, 6],
                  [7, 3, 1, 8],
                  [8, 4, 1, 10],
                  [1, 2, 3, 5],
                  [1, 2, 3, 6]])

    Lb, Ub, idxb = getRowEchelonForm(Test_B)
    print("L =\n", Lb)
    print("U =\n", Ub)
    print("A = LU = \n", Lb @ Ub)

    print("Test C mxn, n>m>r")
    Test_C = np.array([[2, 2, 2, 2],
                       [4, 4, 4, 4],
                       [6, 18, 22, 12]])

    Lc, Uc, idxc = getRowEchelonForm(Test_C)
    print("L =\n", Lc)
    print("U =\n", Uc)
    print("A = LU = \n", Lc @ Uc)