import numpy as np
from LUFactorization import getRowEchelonForm
from OrthogonalReduction import HouseholderReduction, GivensReduction
from QRFactorization import GramSchmidt
from URVFactorization import URVFactorization


class LinearSystem:
    """
    A: 1. mxn 的矩阵
       2. int 随机生成一个 AxA 的矩阵
       3. (m, n) 随机生成一个 m x n 的矩阵
    b:
    """
    def __init__(self, A=None, b=None, random=(0, 0)):
        if random == (0, 0):
            self.A = A.astype(np.float64)
            self.b = b.astype(np.float64)
        else:
            random_matrix = np.random.random(random)
            random_matrix = np.array(random_matrix)
            self.A = random_matrix[:, :-1].astype(np.float64)
            self.b = random_matrix[:, [-1]].astype(np.float64)

        self.m, self.n = self.A.shape

        self.Lower = None
        self.Upper = None
        self.index = None

        self.Q = None
        self.R = None

        self.urv_u = None
        self.urv_r = None
        self.urv_v = None

    # 查看是否有解 doesn't matter
    def isSolvable(self):
        row_len = self.A.shape[0]
        col_len = self.A.shape[1]
        # 统计零行
        zero_row = 0
        zero_aug_row = 0
        for row in range(row_len):
            # 统计零元素个数
            zero = np.sum(np.where(self.A[row][:-1], 0, 1))
            if zero == col_len - 1:
                if self.A[row][col_len - 1] == 0:
                    zero_aug_row += 1
                zero_row += 1

        rank_A = row_len - zero_row
        rank_Ab = row_len - zero_aug_row
        print(f"Rank(A) = {rank_A}, Rank(A|b) = {rank_Ab}")
        if rank_A < rank_Ab:
            print("is not solvable")
        else:
            if rank_A < col_len - 1:
                print("infinite solutions")
            elif rank_A == col_len - 1:
                print("solvable")
            else:
                print("is not solvable")

    # LU 分解
    def LU_Factorization(self):
        L, U, index = getRowEchelonForm(self.A)
        self.Lower = L
        self.Upper = U
        self.index = index

        print("L矩阵:\n", self.Lower)
        print("U矩阵:\n", self.Upper)
        print("LU矩阵:(LU分解过程中会有行交换的行为，所以还原的矩阵可能会与原矩阵不对应\n", self.Lower @ self.Upper)

    # 使用 LU分解 解线性方程组
    # Ly = b
    # Ux = y
    # A(mxn) = L(mxm)U(mxn)
    def solve_by_LU(self):
        if self.Lower is not None:
            # 根据 index 交换b的顺序
            b_sorted = [self.b[self.index.index([i])] for i in range(len(self.index))]
            # 求解 y
            y = []
            for i in range(self.m):
                bi = b_sorted[i].copy()
                for j in range(i):
                    bi -= y[j] * self.Lower[i, j]
                yi = bi
                y.append(yi)
            # 求解 x
            x = [0] * self.m
            for i in range(min(self.m, self.n) - 1, -1, -1):
                yi = y[i]
                for j in range(min(self.m, self.n) - 1, i, -1):
                    yi -= x[j] * self.Upper[i, j]
                xi = yi / self.Upper[i, i] if self.Upper[i, i] != 0 else 0
                x[i] = xi

            print("求解x=")
            for i in range(len(x)):
                print(f"x{i} = {x[i]}")
        else:
            print("LU factorization first")

    # QR 分解
    def QR_Factorization(self, methods="gm"):
        if methods == "gm":
            self.Q, self.R = GramSchmidt(Matrix=self.A)

        elif methods == "householder":
            self.Q, self.R = HouseholderReduction(Matrix=self.A)

        elif methods == "givens":
            self.Q, self.R = GivensReduction(Matrix=self.A)

        print("Q矩阵:\n", self.Q)
        print("R矩阵:\n", self.R)

    # 使用 QR分解 解线性方程组
    # Ax = QRx = b
    # Rx = Q.t @ b
    def solve_by_QR(self):
        b = self.Q.T @ self.b
        # 求解 x
        x = [0] * self.m
        for i in range(min(self.m, self.n) - 1, -1, -1):
            yi = b[i]
            for j in range(min(self.m, self.n) - 1, i, -1):
                yi -= x[j] * self.R[i, j]
            xi = yi / self.R[i, i] if self.R[i, i] != 0 else 0
            x[i] = xi

        print(x)

    # 计算方阵的行列式
    def get_determinant(self):
        if self.m != self.n:
            print("非方阵不具有行列式")
        if self.Lower is None:
            self.LU_Factorization()

        determinant = 1
        for i in range(self.m):
            determinant = determinant * self.Lower[i, i] * self.Upper[i, i]

        print(f"该方阵行列式 = {determinant}")

    def URV_Factorization(self):
        self.urv_u, self.urv_r, self.urv_v = URVFactorization(Matrix=self.A)
        print("U矩阵\n", self.urv_u)
        print("R矩阵\n", self.urv_r)
        print("V矩阵\n", self.urv_v)
        print("URV矩阵\n", self.urv_u @ self.urv_r @ self.urv_v)


if __name__ == "__main__":
    # 取消科学记数法显示
    np.set_printoptions(suppress=True)
    """A_test = np.array([[0, -20, -14],
                  [3, 27, -4],
                  [4, 11, -2]])
    x_true = np.array([[x] for x in range(3)])
    b_test = np.array([row @ x_true for row in A_test])
    print(b_test)
    # 线性系统初始化
    mat = LinearSystem(A=A_test, b=b_test)
    # 对矩阵使用LU分解
    mat.LU_Factorization()
    # 求解分解后的线性系统
    mat.solve_by_LU()

    # 对矩阵使用QR分解
    mat.QR_Factorization(methods="givens")
    # 求解分解后的线性系统
    mat.solve_by_QR()
    # 求解方阵行列式
    mat.get_determinant()"""

    msg = input("请选择一个线性系统(1->输入data.txt文件 2->随机生成一个mxn矩阵 0->退出)\n")
    while msg != "0":
        if msg == "1":
            A = []
            b = []
            with open("data.txt", "r") as f:
                line = f.readline().strip("\n")
                while line:
                    if line == "A":
                        m, n = f.readline().strip("\n").split(",")
                        A = []
                        for row in range(int(m)):
                            A.append([int(x) for x in f.readline().strip("\n").split(" ")])

                    elif line == "b":
                        b = [[int(x)] for x in f.readline().strip("\n").split(" ")]

                    elif line == "x":
                        true_x = [[int(x)] for x in f.readline().strip("\n").split(" ")]

                    line = f.readline().strip("\n")
            A = np.array(A)
            b = np.array(b)
            mat = LinearSystem(A=A, b=b)

        elif msg == "2":
            shape = input("请输入矩阵长宽 e.g. 3,3\n")
            m, n = shape.split(",")
            m, n = int(m), int(n)
            mat = LinearSystem(random=(m, n+1))

        else:
            msg = input("1 输入data.txt文件\n2 随机生成一个mxn矩阵\n0 退出\n")
            continue

        msg_2 = input("请选择分解方法(1->LU 2->QR 3->URV 4->求行列式 "
                      "5->使用LU求解线性系统 6->使用QR求解线性系统 0->退出)\n")
        while msg_2 != "0":
            if msg_2 == "1":
                mat.LU_Factorization()
            elif msg_2 == "2":
                method = input("请选择QR分解算法(gm->gram-schmidt "
                               "householder->Householder "
                               "reduction givens->Givens reduction)\n")
                mat.QR_Factorization(methods=method)
            elif msg_2 == "3":
                mat.URV_Factorization()
            elif msg_2 == "4":
                mat.get_determinant()
            elif msg_2 == "5":
                mat.solve_by_LU()
            elif msg_2 == "6":
                mat.solve_by_QR()
            else:
                msg_2 = input("1->LU \n2->QR \n3->URV \n4->求行列式 "
                              "\n5->使用LU求解线性系统 \n6->使用QR求解线性系统 \n0->退出\n")
                continue

            msg_2 = input("请选择分解方法(1->LU 2->QR 3->URV 4->求行列式 "
                          "5->使用LU求解线性系统 6->使用QR求解线性系统 0->退出)\n")

        msg = input("请选择一个线性系统(1->输入data.txt文件 2->随机生成一个mxn矩阵 0->退出)\n")

    print("程序运行结束！")