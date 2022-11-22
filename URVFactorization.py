import numpy as np
from QRFactorization import HouseholderReduction, GivensReduction, GramSchmidt

def URV(Matrix):
    r = np.linalg.matrix_rank(Matrix)
    mat1 = Matrix[:, :r]
    mat2 = Matrix[:, r:]
    Q1, R1 = GramSchmidt(mat1)
    print(Q1)
    Q2, R2 = GramSchmidt(mat2)
    U = np.hstack((Q1, Q2))

    mat3 = Matrix.T[:, :r]
    mat4 = Matrix.T[:, r:]
    Q3, R3 = GramSchmidt(mat3)
    Q4, R4 = GramSchmidt(mat4)
    V = np.hstack((Q3, Q4))

    R = U.T @ Matrix @ V

    return U, R, V

def URVFactorization(Matrix):
    """
    URV 分解
    :param Matrix:
    :return: U, R, V
    """
    Q, R = GivensReduction(Matrix)
    r = np.linalg.matrix_rank(Matrix)
    mat1 = R[0:r, :]
    Q1, R1 = GivensReduction(mat1.T)
    U = Q
    V = Q1
    R1 = np.dot(np.dot(U.T, Matrix), V)

    return U, R1, V

if __name__ == "__main__":
    # 取消科学记数法显示
    np.set_printoptions(suppress=True)
    A = np.array([[1, 2, 3, 4],
                  [2, 4, 6, 8],
                  [3, 6, 9, 12]])

    U, R, V = URV(A)
    print(U)
    print(R)
    print(V)
