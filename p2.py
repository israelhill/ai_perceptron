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


# returns array of data points and a second array of the class each data point belongs to
def get_data():
    classes = []
    data = {}
    p_l = []
    p_w = []
    with open('irisdata.csv', 'rb') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
        for row in reader:
            species = row[0].split(',')[4]
            petal_length = row[0].split(',')[2]
            petal_width = row[0].split(',')[3]
            if species == 'versicolor':
                classes.append(-1)
                p_l.append(float(petal_length))
                p_w.append(float(petal_width))
            elif species == 'virginica':
                classes.append(1)
                p_l.append(float(petal_length))
                p_w.append(float(petal_width))
        data['petal_length'] = p_l
        data['petal_width'] = p_w
    final_data = []
    for i in range(0, len(data['petal_length'])):
        final_data.append([data['petal_length'][i], data['petal_width'][i]])
    return final_data, classes


def plot_petal_length_width(weights, equation):
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

    x = np.arange(100)
    y = []
    for i in range(0, 100):
        y.append(equation(weights[0],weights[1],weights[2], x[i]))
    plt.plot(x, y)
    plt.show()


def classify(x1, x2, boundary_formula):
    val = boundary_formula(x1, x2)
    if val >= 0:
        return 'virginica', val
    else:
        return 'versicolor', val


def mse(data_points, weights, classes):
    total = 0
    flower_data = np.array(data_points).astype(float)
    for i in range(len(data)):
        dot_product = weights[0] + np.dot(weights[1:], flower_data[i])
        sq_difference = math.pow((classify_prediction(dot_product) - classes[i]), 2)
        total += sq_difference
    return total


def update_w(old_weights, step_size, data_points, data_point_class, dimension):
    total = 0
    errors = 0
    for i in range(0, len(data_points)):
        dot_product = old_weights[0] + np.dot(old_weights[1:], data_points[i])
        # print('Pred: ' + str(classify_prediction(dot_product)) + '    Actual: ' + str(data_point_class[i]))
        difference = (classify_prediction(dot_product)-data_point_class[i])
        if dimension > 0:
            multiplier = data_points[i][dimension-1]
        else:
            multiplier = 1
        total += difference * multiplier
        if difference != 0:
            errors += 1
    # print('Errors: ' + str(errors))
    return total * step_size


def g_descent(iterations, weights, step_size, data_points, classes):
    for i in range(0, iterations):
        weights[0] -= update_w(weights, step_size, data_points, classes, 0)
        weights[1] -= update_w(weights, step_size, data_points, classes, 1)
        weights[2] -= update_w(weights, step_size, data_points, classes, 2)
        # print('w0: ' + str(weights[0]) + ' | ' + 'w1: ' + str(weights[1]) + ' | ' + 'w2: ' + str(weights[2]))
        # print(mse(data_points, weights, classes))


def classify_prediction(val):
    # print(val)
    if val >= 0:
        return 1
    else:
        return -1

# def make_summed_gradient_plot(data_points, weights):
#     updated_weights = []
#     for i in range(0, 100):
#         for x in range(0, len(data_points)):


def plot_extra_credit(weights, centers):
    plt.title('Extra Credit: Circle Decision Boundary')
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.gca().set_aspect('equal')
    # plt.scatter(setosa['petal_length'], setosa['petal_width'])
    plt.scatter(versicolor['petal_length'], versicolor['petal_width'], c='red')
    plt.scatter(virginica['petal_length'], virginica['petal_width'], c='b', marker='+')
    plt.legend(['Versicolor', 'Virginica'], loc='upper left')

    x = np.linspace(-1, 8.0, 100)
    y = np.linspace(-1, 8.0, 100)
    x_mesh, y_mesh = np.meshgrid(x, y)
    w0 = weights[0]
    w1 = weights[1]
    w2 = weights[2]
    c0 = centers[0]
    c1 = centers[1]
    z = w1 * ((x_mesh - c0) ** 2) + w2 * ((y_mesh - c1) ** 2) + w0
    plt.contour(x_mesh, y_mesh, z, [0])
    plt.show()


def plot_mse(data_points, weights, step_size, classes):
    iterations = 1000
    # x = []
    x = range(0, 10)
    y = []
    count = 0
    last_y = (2**63) - 1
    for i in range(0, iterations):
        weights[0] -= update_w(weights, step_size, data_points, classes, 0)
        weights[1] -= update_w(weights, step_size, data_points, classes, 1)
        weights[2] -= update_w(weights, step_size, data_points, classes, 2)
        # print('w0: ' + str(weights[0]) + ' | ' + 'w1: ' + str(weights[1]) + ' | ' + 'w2: ' + str(weights[2]))
        error = (mse(data_points, weights, classes))
        # if i % 10 == 0 and error < last_y:
        #     y.append(error)
        #     x.append(count)
        #     count += 1
        #     last_y = error
        if i % 100 == 0:
            y.append(error)
            last_y = error
    print(len(y))
    plt.plot(x, y)
    plt.show()




if __name__ == "__main__":
    data2 = read_csv()

    data, classes = get_data()
    weights = [0, 10, 10]

    plot_mse(data, weights, 0.01, classes)

    # 2B choose good and bad function and plot the boundary line over the data
    # mse = mse(data, [0, -0.95, -0.95], classes)
    # print(mse)
    # plot_weights = [-15.36, 1.562, 4.8]
    # plot_petal_length_width(plot_weights, lambda w0, w1, w2, x: -(w1 * x + w0) / w2)
    # g_descent(10000, weights, 0.01, data, classes)
    # print(classify(8, 5, lambda x1, x2: (-0.68*x1 + -0.68*x2) + 8.6))

    # -15.28, 1.56, 4.87
    # centers: 4.0, 1.0
    # circle weights: -0.6, 0.5, 0.5
    # lambda x1, x2, w0, w1, w2, c0, c1: w1 * ((x1 - c0) ** 2) + w2 * ((x2 - c1) ** 2) + w0
    # plot_extra_credit([-0.6, 0.5, 0.5], [4.0, 1.0])
