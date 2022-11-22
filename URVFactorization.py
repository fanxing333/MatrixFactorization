import numpy as np
from QRFactorization import HouseholderReduction, GivensReduction, GramSchmidt

def URVFactorization(Matrix):
    m, n = Matrix.shape
    r = np.linalg.matrix_rank(Matrix)
    P, R1 = HouseholderReduction(Matrix)  # P 就是矩阵 U
    B = (P.T @ Matrix)[:r, :]
    Q, T = HouseholderReduction(B.T)  # Q.T 就是矩阵 V
    R = np.pad(T.T,  ((0, m-r), (0, 0)), 'constant', constant_values=(0, 0))
    #R = P.T @ Matrix @ Q
    #print(P.T@Matrix)

    return P, R, Q.T

if __name__ == "__main__":
    # 取消科学记数法显示
    np.set_printoptions(suppress=True)
    As = np.array([[1, 2, 3, 4],
                  [2, 4, 6, 8],
                  [3, 6, 9, 13],
                  [3, 6, 9, 13],
                  [3, 6, 9, 13]])
    A = np.array([[0, -20, -14],
                  [3, 27, -4],
                  [4, 11, -2],
                  [4, 11, -2]])

    print("Test A mxm, r=m")
    Test_A = np.array([[0, -20, -14],
                          [3, 27, -4],
                          [4, 11, -2]])
    Ua, Ra, Va = URVFactorization(Test_A)
    print("U矩阵\n", Ua)
    print("R矩阵\n", Ra)
    print("V矩阵\n", Va)
    print("URV矩阵\n", Ua @ Ra @ Va)

    print("Test B mxm, r=n<m")
    Test_B = np.array([[1, 2, 3],
                       [2, 4, 6],
                       [1, 1, -14]])
    Ub, Rb, Vb = URVFactorization(Test_B)
    print("U矩阵\n", Ub)
    print("R矩阵\n", Rb)
    print("V矩阵\n", Vb)
    print("URV矩阵\n", Ub @ Rb @ Vb)

    print("Test C mxn, m<n r=2")
    Test_C = np.array([[-4, -2, -4, -2],
                       [2, -2, 2, 1],
                       [-4, 1, -4, -2]])
    Uc, Rc, Vc = URVFactorization(Test_C)
    print("U矩阵\n", Uc)
    print("R矩阵\n", Rc)
    print("V矩阵\n", Vc)
    print("URV矩阵\n", Uc @ Rc @ Vc)

    print("Test D mxn, m>n r=2")
    Test_D = np.array([[-4, -2, -4],
                       [2, 1, 2],
                       [-4, -2, -4],
                       [-3, 2, -1]])
    Ud, Rd, Vd = URVFactorization(Test_D)
    print("U矩阵\n", Ud)
    print("R矩阵\n", Rd)
    print("V矩阵\n", Vd)
    print("URV矩阵\n", Ud @ Rd @ Vd)