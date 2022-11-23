import numpy as np

def ModifiedGramSchmidt(Matrix):
    m, n = Matrix.shape
    Matrix = Matrix.astype(np.float64)

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


if __name__ == "__main__":
    # 取消科学记数法显示
    np.set_printoptions(suppress=True)

    print("Test A mxm")
    TestA = np.array([[0, -20, -14],
                      [3, 27, -4],
                      [4, 11, -2]])

    Qa, Ra = GramSchmidt(TestA)
    print("Q矩阵\n", Qa)
    print("R矩阵\n", Ra)
    print("QR矩阵\n", Qa @ Ra)

    print("Test A mxn")
    print("Test B, mxn and m>n")
    TestB = np.array([[0, -20, -14],
                      [3, 27, -4],
                      [4, 11, -2],
                      [2, 5, 6]])
    Qb, Rb = GramSchmidt(TestB)
    print("Q矩阵4x3\n", Qb)
    print("R矩阵3x3\n", Rb)
    print("QR矩阵\n", Qb @ Rb)