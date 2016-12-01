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


def simple_threshold(x, y, boundary_formula):
    if boundary_formula(x) <= y:
        print('virginica')
        return 'virginica'
    else:
        print('versicolor')
        return 'versicolor'


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

    error = 0
    for i in range(0, len(data1['petal_length'])-1):
        print(data1['petal_length'][i])
        print(data1['petal_width'][i])
        if simple_threshold(float(data1['petal_length'][i]), float(data1['petal_width'][i])) != class1:
            error += 1
        else:
            error += 0

    for i in range(0, len(data2['petal_length'])-1):
        if simple_threshold(float(data2['petal_length'][i]), float(data2['petal_width'][i])) != class2:
            error += 1
        else:
            error += 0

    return float(error)/(len(data1['petal_length']) + len(data2['petal_length']))*4

if __name__ == "__main__":
    data = read_csv()
    # simple_threshold(5.1, 2)
    # mse = mse(data, 1, ['versicolor', 'virginica'])
    # print(mse)
    plot_petal_length_width(lambda x: -0.68*x+5)