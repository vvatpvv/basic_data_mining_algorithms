import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

def get_data():
    digits = datasets.load_digits()
    X = digits.images
    Y = digits.target
    X_0 = X[(Y == 0)]
    Y_0 = [0 for i in range(X_0.shape[0])]
    X_1 = X[(Y == 1)]
    Y_1 = [1 for i in range(X_1.shape[0])]

    X_new = np.concatenate((X_0, X_1), axis=0).reshape((360, 64))
    Y_new = np.concatenate((Y_0, Y_1), axis=0)
    # print(X_new.shape)
    # print(X_new)
    return X_new, Y_new


def calculate_PCs():
    X, Y = get_data()

    X_mean = np.mean(X, axis=0)
    X_c = X - X_mean
    u, s, v = np.linalg.svd(X_c, full_matrices=False, compute_uv=True)
    X_pca = np.dot(u, np.diag(s))
    PC1 = - X_pca[:, 0]
    PC2 = X_pca[:, 1]

    return PC1, PC2, Y

def draw_graph():
    PC1, PC2, Y = calculate_PCs()
    X_0_PC1 = PC1[(Y == 0)]
    X_0_PC2 = PC2[(Y == 0)]
    plt.scatter(X_0_PC1, X_0_PC2, marker='^', c="r")

    X_1_PC1 = PC1[(Y == 1)]
    X_1_PC2 = PC2[(Y == 1)]
    plt.scatter(X_1_PC1, X_1_PC2, marker='o', c="b")

    plt.show()




if __name__ == "__main__":
    # get_data()
    draw_graph()