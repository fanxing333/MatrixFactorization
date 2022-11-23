import numpy as np
import os
from main import LinearSystem

if __name__ == "__main__":
    """
    A_list = []
    b_list = []
    true_x_list = []
    
    with open("data.txt", "r") as f:
        line = f.readline().strip("\n")
        while line:
            if line == "A":
                m, n = f.readline().strip("\n").split(",")
                A = []
                for row in range(int(m)):
                    A.append([int(x) for x in f.readline().strip("\n").split(" ")])
                A_list.append(np.array(A))

            elif line == "b":
                b = [[int(x)] for x in f.readline().strip("\n").split(" ")]
                b_list.append(np.array(b))

            elif line == "x":
                true_x = [[int(x)] for x in f.readline().strip("\n").split(" ")]
                true_x_list.append(np.array(true_x))

            line = f.readline().strip("\n")

    for i in range(len(A_list)):
        mat = LinearSystem(A=A_list[i], b=b_list[i])
        mat.LU_Factorization()
        mat.solve_by_LU()
        print("答案是\n", true_x_list[i])
    """
    print("-----------------开始LU分解测试-------------------")
    lu_test = "python LUFactorization.py"
    print(os.system(lu_test))
    print("-----------------结束LU分解测试-------------------")

    print("-----------------开始QR分解测试-------------------")
    qr_test = "python QRFactorization.py"
    print(os.system(qr_test))
    print("-----------------结束QR分解测试-------------------")

    print("-----------------开始正交约减测试-------------------")
    or_test = "python OrthogonalReduction.py"
    print(os.system(or_test))
    print("-----------------结束正交约减测试-------------------")

    print("-----------------开始URV分解测试-------------------")
    urv_test = "python URVFactorization.py"
    print(os.system(urv_test))
    print("-----------------结束URV分解测试-------------------")
