import scipy.io
import math
import operator

# Calculate distance
def calcDistance(x1, x2):
    distance = 0
    for i in range(len(x1)):
        distance += pow((float(x1[i]) - float(x2[i])), 2)
    return math.sqrt(distance)

# Find center
# Input: list in list
# Return list which is coordinate of center
def findCenter(set_x):
    center = []
    for j in range(len(set_x[0])):
        sum_i = 0
        for i in range(len(set_x)):
            sum_i += set_x[i][j]
        center.append(sum_i/len(set_x))
    return center

# Find all centers
# Return list of [center, class]
def findAllCenters(trainSet_x, trainSet_y, classes):
    centers = []
    for temp in range(classes):
        x = temp + 1
        current_trainSet = []
        for i in range(len(trainSet_y)):
            if trainSet_y[i] == x:
                current_trainSet.append(trainSet_x[i])
        centers.append([findCenter(current_trainSet), x])
    return centers

def Centroid(classes, X_train, y_train, X_test, y_test):
    predictions = []
    all_centers = findAllCenters(X_train, y_train, classes)
    for inst in range(len(y_test)):
        distances_from_centers = []
        for center in all_centers:
            dist = calcDistance(X_test[inst], center[0])
            distances_from_centers.append([dist, center[1]])
        distances_from_centers.sort(key=operator.itemgetter(0))
        predictions.append(distances_from_centers[0][1])

    for i in range(len(predictions)):
        if predictions[i] < 10:
            print("", end=" ")
        print(predictions[i], end=" ")
        if (i+1) % 27 == 0:
            print("")
    print("")

    # Evaluate accuracy
    count_correct = 0
    count_incorrect = 0
    for i in range(len(y_test)):
        if y_test[i] == predictions[i]:
            count_correct += 1
        else:
            count_incorrect += 1
    print('Accuracy: ' + str(count_correct/(count_incorrect + count_correct)*100) + '%')

# Load train and test data
# X_train and X_test are list in list, y_train and y_test are list
X_train = list(map(list, zip(*scipy.io.loadmat("trainX.mat")['trainX'])))
y_train = scipy.io.loadmat("trainY.mat")['trainY'][0]
X_test = list(map(list, zip(*scipy.io.loadmat("testX.mat")['testX'])))
y_test = scipy.io.loadmat("testY.mat")['testY'][0]

# centroid = Centroid(40, X_train, y_train, X_test, y_test)
centroid = Centroid(26, X_train, y_train, X_test, y_test)
