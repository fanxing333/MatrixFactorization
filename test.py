import numpy as np

from main import LinearSystem

if __name__ == "__main__":
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