import numpy as np

def ModifiedGramSchmidt(Matrix):
    m, n = Matrix.shape
    Matrix = Matrix.astype(np.float64)
    if n > m:
        raise ValueError

    # for k = 1
    U = [Matrix[:, [i]] for i in range(n)]
    U[0] = U[0] / np.linalg.norm(U[0])
    # for k > 1
    for k in range(1, n):
        # for j in (k, n)
        for j in range(k, n):
            U[j] = U[j] - (U[k-1].T @ U[j])[0] * U[k-1]
        U[k] = U[k] / np.linalg.norm(U[k])

    return U

def GramSchmidt(Matrix):
    """
    Gram-Schmidt 方法QR分解
    :param Matrix:
    :return: Q R
    """
    m, n = Matrix.shape
    Matrix = Matrix.astype(np.float64)
    if n > m:
        raise ValueError

    Q = Matrix.copy()
    R = np.zeros((n, n))
    # for k = 1
    v1 = np.linalg.norm(Matrix[:, [0]])
    q1 = Matrix[:, [0]] / v1

    R[0, 0] = v1
    Q[:, [0]] = q1
    if n == 1:
        return Q, R
    # for k > 1
    for k in range(1, n):
        for j in range(k):
            R[j, k] = Q[:, [j]].T @ Matrix[:, [k]]
            Q[:, [k]] = Q[:, [k]] - R[j, k] * Q[:, [j]]

        R[k, k] = np.linalg.norm(Q[:, [k]])
        Q[:, [k]] = Q[:, [k]] / R[k, k]

    return Q, R

def get_augmented_matrix(Matrix, target_dim):
    """
    对 nxn 的矩阵向左上增广，对角元素为1，其余元素为0，直到满足目标维度
    :param Matrix: np.array([[1,2],
                             [2,3]])
    :param target_dim: 3
    :return: np.array([[1,0,0]
                       [0,1,2],
                       [0,2,3]])
    """
    m, n = Matrix.shape
    if m == target_dim:
        return Matrix

    gap_dim = target_dim - m

    Matrix = np.pad(Matrix, ((gap_dim, 0), (gap_dim, 0)), 'constant', constant_values=(0, 0))\
             + np.pad(np.identity(gap_dim), ((0, m), (0, m)), 'constant', constant_values=(0, 0))

    return Matrix

def HouseholderReduction(Matrix):
    """
    使用 householder 约减实现矩阵A的QR分解 A(mxn) = Q(mxm)R(mxn)
    :param Matrix: A mxn
    :return: 正交矩阵 Q mxm, 上三角矩阵 R mxn
    """
    m, n = Matrix.shape
    Q_list = []
    Matrix_copy = Matrix.copy()
    for i in range(min(m-1, n)):

        I = np.identity(m - i)
        e1 = I[:, [0]]

        u1 = Matrix_copy[:, [0]] - np.linalg.norm(Matrix_copy[:, [0]]) * e1
        R1 = I - 2 * (u1 @ u1.T) / (u1.T @ u1)

        Matrix_copy = (R1 @ Matrix_copy)[1:, 1:]
        R1 = get_augmented_matrix(R1, m)
        Q_list.append(R1)

    # Q_inv @ A = R
    Q_inv = np.identity(m)
    for q in Q_list:
        Q_inv = q @ Q_inv

    R = Q_inv @ Matrix
    return Q_inv.T, R

def GivensReduction(Matrix):
    """
    使用 Givens Reduction 进行矩阵 A 的 QR分解，实现思路和householder类似，中间操作改一下就行
    :param Matrix:
    :return:
    """
    m, n = Matrix.shape
    Q_list = []
    Matrix_copy = Matrix.copy()
    # 外层循环，对子矩阵的每一列进行plane rotation
    for i in range(m - 1):
        I = np.identity(m - i)
        #print(Matrix_copy)
        col_i = Matrix_copy[:, [0]]
        # 对第 i 列进行 plane rotation
        col = col_i
        Pij_list = []
        for j in range(1, m - i):
            Pij = plane_rotation(col, 0, j)
            col = (Pij @ col)
            Pij_list.append(Pij)

        Pi = I
        for p in Pij_list:
            Pi = p @ Pi

        Matrix_copy = (Pi @ Matrix_copy)[1:, 1:]
        Pi = get_augmented_matrix(Pi, m)
        Q_list.append(Pi)

    Q_inv = np.identity(m)
    for q in Q_list:
        Q_inv = q @ Q_inv
    # P @ A = R
    return Q_inv.T, Q_inv @ Matrix

def plane_rotation(x, i, j):
    """
    对一个向量 x 进行旋转操作使得 第 i 个元素为，第 j 个元素为 0，返回变换矩阵 P
    :param x:
    :param i:
    :param j:
    :return:
    """
    P = np.identity(len(x))

    c = x[i, 0] / np.sqrt(np.power(x[i, 0], 2) + np.power(x[j, 0], 2))
    s = x[j, 0] / np.sqrt(np.power(x[i, 0], 2) + np.power(x[j, 0], 2))
    P[i, i], P[j, j] = c, c
    P[i, j] = s
    P[j, i] = -s

    return P


if __name__ == "__main__":
    # 取消科学记数法显示
    np.set_printoptions(suppress=True)
    A = np.array([[0, -20, -14],
                  [3, 27, -4],
                  [4, 11, -2]])
    """
    A2 = np.array([[1, 19, -34],
                   [-2, -5, 20],
                   [2, 8, 37]])
    """
    """
    random_matrix = np.random.random((10, 10))
    random_matrix = np.array(random_matrix)
    print(random_matrix)
    Q, R = HouseholderReduction(random_matrix)
    print(Q)
    print(R)
    print(Q @ R)
    Q, R = GivensReduction(random_matrix)
    print(Q)
    print(R)
    print(Q @ R)
    
    AA = np.array([[1, 2, 3],
                   [2, 4, 6],
                   [3, 6, 9],
                   [3, 6, 9]])
    Qg, Rg = GramSchmidt(AA)
    print(Qg)
    print(Rg)"""

    print("Test A, mxm")
    TestA = np.array([[0, -20, -14],
                      [3, 27, -4],
                      [4, 11, -2]])
    Qa, Ra = HouseholderReduction(TestA)
    print("Q矩阵\n", Qa)
    print("R矩阵\n", Ra)
    print("QR矩阵\n", Qa @ Ra)

    print("Test B, mxn and m>n")
    TestB = np.array([[0, -20, -14],
                      [3, 27, -4],
                      [4, 11, -2],
                      [2, 5, 6]])
    Qb, Rb = HouseholderReduction(TestB)
    print("Q矩阵4x4\n", Qb)
    print("R矩阵4x3\n", Rb)
    print("QR矩阵\n", Qb @ Rb)

    print("Test C, mxn and m<n")
    TestC = np.array([[0, -20, -14, 4],
                      [3, 27, -4, 3],
                      [4, 11, -2, 2]])
    Qc, Rc = HouseholderReduction(TestC)
    print("Q矩阵3x3\n", Qc)
    print("R矩阵3x4\n", Rc)
    print("QR矩阵\n", Qc @ Rc)

    print("Test D, mxm and r=1")
    TestD = np.array([[1, -20, -14],
                      [1, -20, -14],
                      [1, -20, -14]])
    Qd, Rd = HouseholderReduction(TestD)
    print("Q矩阵3x3\n", Qd)
    print("R矩阵3x3\n", Rd)
    print("QR矩阵\n", Qd @ Rd)