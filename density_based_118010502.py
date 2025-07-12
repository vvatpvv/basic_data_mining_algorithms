import scipy.io
import math
import operator

# Calculate distance
# Return root((x1[0]-x2[0])^2 + ... + (x1[length-1]-x2[length-1])^2)
def calcDistance(x1, x2):
    distance = 0
    for i in range(len(x1)):
        distance += pow((float(x1[i]) - float(x2[i])), 2)
    return math.sqrt(distance)

# Find neighbors that are within distance of D
# Return a list consisting of [trainSet_x[i], trainSet_y[i]]
def findNeighbors(trainSet_x, trainSet_y, testInstance, D):
    neighbors = []
    exist = 0
    radius = D
    while exist == 0:
        for x in range(len(trainSet_y)):
            dist = calcDistance(testInstance, trainSet_x[x])
            if dist <= radius:
                neighbors.append([trainSet_x[x], trainSet_y[x]])
                exist = 1
            else:
                continue
        # radius += 100
        radius += 0.5
    return neighbors

# Find class in which neighbors are most often found
def nearestNeighbor(neighbors):
    classCounter = {}
    for x in range(len(neighbors)):
        neighborClass = neighbors[x][1]
        if neighborClass in classCounter:
            classCounter[neighborClass] += 1
        else:
            classCounter[neighborClass] = 1
    finalCounter = sorted(classCounter.items(), key=operator.itemgetter(1), reverse=True)
    return finalCounter[0][0]

def DensityBased(distance, X_train, y_train, X_test, y_test):
    predictions = []
    for inst in range(len(y_test)):
         neighbors = findNeighbors(X_train, y_train, X_test[inst], distance)
         nearestClass = nearestNeighbor(neighbors)
         predictions.append(nearestClass)

    for i in range(len(predictions)):
        if predictions[i] < 10:
            print("", end=" ")
        print(predictions[i], end=" ")
        if (i + 1) % 27 == 0:
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

# densitybased = DensityBased(600, X_train, y_train, X_test, y_test)
densitybased = DensityBased(10, X_train, y_train, X_test, y_test)
