import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap

# X = []
Y = []
for i in range(100):
    Y.append(-1)
for i in range(100):
    Y.append(1)
# for i in range(80):
#     m = random.randint(0, 200)
#     n = random.randint(0, 200)
#     X.append([min(m, n), max(m, n) - min(m, n)])
# for i in range(40):
#     m = random.randint(190, 210)
#     n = random.randint(0, 200)
#     X.append([min(m, n), max(m, n) - min(m, n)])
# for i in range(80):
#     m = random.randint(0, 200)
#     n = random.randint(0, 200)
#     X.append([200 - min(m, n), 200 - max(m, n) + min(m, n)])

X = [[69, 114], [43, 101], [44, 42], [15, 172], [46, 87], [8, 91], [110, 7], [90, 43], [17, 104], [98, 5], [23, 176], [90, 64], [11, 73], [64, 109],
     [84, 48], [2, 85], [51, 67], [163, 19], [27, 119], [57, 32], [31, 28], [25, 68], [136, 60], [191, 4], [95, 17], [179, 21], [4, 154], [122, 67],
     [104, 61], [39, 15], [47, 29], [35, 121], [19, 14], [69, 69], [1, 142], [90, 3], [82, 118], [98, 50], [37, 114], [36, 153], [43, 150], [81, 79],
     [55, 59], [137, 54], [49, 41], [24, 129], [100, 72], [163, 2], [13, 15], [170, 10], [44, 106], [34, 47], [43, 130], [82, 97], [78, 81],
     [137, 20], [24, 108], [103, 43], [100, 56], [82, 33], [124, 47], [61, 7], [109, 37], [12, 188], [108, 58], [47, 28], [121, 71], [35, 138],
     [53, 115], [15, 136], [83, 70], [133, 20], [168, 31], [81, 29], [59, 67], [38, 62], [108, 10], [130, 53], [181, 19], [110, 21], [42, 168],
     [65, 132], [95, 100], [121, 88], [25, 185], [137, 67], [58, 147], [47, 161], [29, 177], [199, 1], [74, 132], [160, 35], [111, 98], [197, 3],
     [98, 94], [70, 131], [89, 119], [34, 167], [1, 199], [161, 46], [85, 119],[155, 43], [23, 174], [32, 158], [12, 193], [156, 40], [187, 5],
     [60, 135], [56, 142], [182, 19], [103, 93], [178, 25], [185, 15], [161, 47], [40, 154], [120, 73], [119, 71], [80, 128], [160, 44], [88, 114],
     [132, 134], [23, 196], [169, 47], [34, 194], [82, 168], [95, 140], [155, 82], [190, 45], [184, 155], [88, 187], [134, 95], [77, 146], [190, 40],
     [75, 193], [9, 195], [124, 92], [45, 165], [65, 151], [152, 155], [164, 97], [195, 25], [44, 199], [149, 122], [171, 64], [23, 181], [185, 93],
     [122, 98], [194, 113], [181, 52], [200, 140], [184, 105], [54, 169], [149, 159], [132, 80], [192, 94], [109, 189], [194, 70], [137, 162],
     [193, 41], [76, 198], [36, 174], [139, 180], [114, 105], [52, 197], [150, 95], [165, 37], [132, 140], [47, 158], [132, 198], [146, 180],
     [172, 171], [53, 148], [192, 188], [32, 199], [113, 174], [167, 67], [94, 191], [107, 190], [73, 158], [200, 135], [107, 107], [152, 180],
     [90, 124], [144, 75], [196, 163], [178, 35], [194, 130], [167, 110], [80, 194], [93, 143], [64, 159], [123, 149], [175, 45], [134, 70],
     [77, 152], [182, 82], [106, 156], [169, 33], [168, 76], [129, 91]]

sc_X = StandardScaler()
X = sc_X.fit_transform(X)

classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X, Y)

X1, X2 = np.meshgrid(np.arange(start=-2, stop=2, step=0.01),
                     np.arange(start=-2, stop=2, step=0.01))

w = classifier.coef_[0]
b = classifier.intercept_[0]
x_points = np.linspace(-2, 2)
y_points = -(w[0] / w[1]) * x_points - b / w[1]
plt.plot(x_points, y_points)    # plot the hyperplane

u_vec = classifier.coef_[0] / (np.sqrt(np.sum(classifier.coef_[0] ** 2)))
margin = 1 / np.sqrt(np.sum(classifier.coef_[0] ** 2))

boundary = np.array(list(zip(x_points, y_points)))
line_positive = boundary + u_vec * margin
plt.plot(line_positive[:, 0], line_positive[:, 1], 'b--', linewidth=1)      # plot f(x) = 1
line_negative = boundary - u_vec * margin
plt.plot(line_negative[:, 0], line_negative[:, 1], 'b--', linewidth=1)      # plot f(x) = -1

plt.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1], s=50,
            facecolors='none', edgecolors='r', alpha=1, marker='s')

for i, j in enumerate(np.unique(Y)):
    plt.scatter(X[Y == j, 0], X[Y == j, 1],
                c=ListedColormap(('yellow', 'orange'))(i), label=j)     # plot the data points
    for pt in range(len(X[Y == j, 0])):
        c = 0
        for check in range(len(classifier.support_vectors_[:, 0])):
            if X[Y == j, 0][pt] == classifier.support_vectors_[:, 0][check] and X[Y == j, 1][pt] == classifier.support_vectors_[:, 1][check]:
                c = 1
        if c == 0:
            plt.scatter(X[Y == j, 0][pt], X[Y == j, 1][pt], s=50,
                        facecolors='none', edgecolors='r', alpha=1)     # red circles for alpha_i=0

plt.title('SVM')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
