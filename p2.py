import numpy as np
import matplotlib.pyplot as plt
import math
import random
import csv

setosa = {}
versicolor = {}
virginica = {}


def read_csv():
    setosa_petal_length = []
    setosa_petal_width = []
    versicolor_petal_length = []
    versicolor_petal_width = []
    virginica_petal_length = []
    virginica_petal_width = []
    with open('irisdata.csv', 'rb') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
        for row in reader:
            species = row[0].split(',')[4]
            petal_length = row[0].split(',')[2]
            petal_width = row[0].split(',')[3]
            if (species == 'setosa'):
                setosa_petal_length.append(petal_length)
                setosa_petal_width.append(petal_width)
            elif (species == 'versicolor'):
                versicolor_petal_length.append(petal_length)
                versicolor_petal_width.append(petal_width)
            elif (species == 'virginica'):
                virginica_petal_length.append(petal_length)
                virginica_petal_width.append(petal_width)
            setosa['petal_length'] = setosa_petal_length
            setosa['petal_width'] = setosa_petal_width
            versicolor['petal_length'] = versicolor_petal_length
            versicolor['petal_width'] = versicolor_petal_width
            virginica['petal_length'] = virginica_petal_length
            virginica['petal_width'] = virginica_petal_width
    return setosa, versicolor, virginica


def plot_petal_length_width(formula):
    plt.title('Iris Petal length x width')
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    axes = plt.gca()
    axes.set_xlim([0, 8])
    axes.set_ylim([0, 3])
    # plt.scatter(setosa['petal_length'], setosa['petal_width'])
    plt.scatter(versicolor['petal_length'], versicolor['petal_width'], c='red')
    plt.scatter(virginica['petal_length'], virginica['petal_width'], c='b', marker='+')
    plt.legend(['Versicolor', 'Virginica'], loc='upper left')

    x = np.arange(10)
    y = formula(x)
    plt.plot(x, y)
    plt.show()


def classify(x, y, boundary_formula):
    if boundary_formula(x) <= y:
        return 'virginica'
    else:
        return 'versicolor'


def new_classify(x1, x2, boundary_formula):
    val = boundary_formula(x1, x2)
    if val >= 0:
        return 'versicolor'
    else:
        return 'virginica'

def mse(data, boundary, classes):
    setosa_data, versicolor_data, virginica_data = data
    # print(setosa_data)
    # print(versicolor_data)
    # print(virginica_data)

    class1 = classes[0]
    class2 = classes[1]

    if class1 == 'versicolor':
        data1 = versicolor_data
    elif class1 == 'virginica':
        data1 = virginica_data
    else:
        data1 = setosa_data

    if class2 == 'versicolor':
        data2 = versicolor_data
    elif class2 == 'virginica':
        data2 = virginica_data
    else:
        data2 = setosa_data

    sum = 0
    for i in range(0, len(data1['petal_length'])-1):
        if classify(float(data1['petal_length'][i]), float(data1['petal_width'][i]), boundary) == class1:
            val1 = 1
        else:
            val1 = -1
        sum += math.pow(val1 - 1, 2)

    for i in range(0, len(data2['petal_length'])-1):
        if classify(float(data2['petal_length'][i]), float(data2['petal_width'][i]), boundary) == class2:
            val2 = 1
        else:
            val2 = -1
        sum += math.pow(val2 - 1, 2)

    return float(sum)/(len(data1['petal_length']) + len(data2['petal_length']))

if __name__ == "__main__":
    data = read_csv()
    print(new_classify(5.1, 1.8, lambda x1, x2: x1*-0.95 + x2*-0.95 + 6.55))

    # 2B choose good and bad function and plot
    # mse = mse(data, lambda x: -0.68*x+5, ['versicolor', 'virginica'])
    # mse = mse(data, lambda x: -0.95 * x + 6.55, ['versicolor', 'virginica'])
    # print(mse)
    # plot_petal_length_width(lambda x: -0.68*x+5)