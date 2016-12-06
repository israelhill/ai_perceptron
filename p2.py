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
            if species == 'setosa':
                setosa_petal_length.append(petal_length)
                setosa_petal_width.append(petal_width)
            elif species == 'versicolor':
                versicolor_petal_length.append(petal_length)
                versicolor_petal_width.append(petal_width)
            elif species == 'virginica':
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


def plot_petal_length_width(weights, equation, title):
    plt.title(title)
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    axes = plt.gca()
    axes.set_xlim([0, 8])
    axes.set_ylim([0, 3])
    plt.scatter(versicolor['petal_length'], versicolor['petal_width'], c='red')
    plt.scatter(virginica['petal_length'], virginica['petal_width'], c='b', marker='+')
    plt.legend(['Versicolor', 'Virginica'], loc='upper left')
    x = np.arange(100)
    y = []
    for i in range(0, 100):
        y.append(equation(float(weights[0]), float(weights[1]), float(weights[2]), float(x[i])))
    plt.plot(x, y)
    plt.show()


def classify(x1, x2, w, boundary_formula):
    val = boundary_formula(x1, x2, w[0], w[1], w[2])
    if val >= 0:
        return 'virginica', val
    else:
        return 'versicolor', val


def circle_classifier(x1, x2, weights, centers, formula):
    w0 = weights[0]
    w1 = weights[1]
    w2 = weights[2]
    c0 = centers[0]
    c1 = centers[1]
    val = formula(x1, x2, w0, w1, w2, c0, c1)
    if val >= 0:
        print ('(' + str(x1) + ',' + str(x2) + '): virginica')
        return 'virginica', val
    else:
        print ('(' + str(x1) + ',' + str(x2) + '): versicolor')
        return 'versicolor', val


def mse(data_points, weights, classes):
    total = 0
    flower_data = np.array(data_points).astype(float)
    for i in range(len(data)):
        dot_product = float(weights[0]) + float(np.dot(weights[1:], flower_data[i]))
        sq_difference = float(math.pow((classify_prediction(dot_product) - classes[i]), 2))
        total += sq_difference
    return float(total)/float((2*len(data)))


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


def g_descent(max_iterations, weights, step_size, data_points, classes):
    last_ten_errors = []
    i = 0
    error = 0
    while i < max_iterations:
        old_w = weights
        weights[0] -= update_w(old_w, step_size, data_points, classes, 0)
        weights[1] -= update_w(old_w, step_size, data_points, classes, 1)
        weights[2] -= update_w(old_w, step_size, data_points, classes, 2)
        error = float(mse(data_points, weights, classes))
        print('w0: ' + str(weights[0]) + ' | ' + 'w1: ' + str(weights[1]) + ' | ' + 'w2: ' + str(weights[2]))
        print(error)
        if i < 10:
            last_ten_errors.append(error)
            has_converged, error_val = converged(last_ten_errors)
            if has_converged:
                print(error_val, weights)
                return error_val, weights
        else:
            last_ten_errors.pop(0)
            last_ten_errors.append(error)
            has_converged, error_val = converged(last_ten_errors)
            if has_converged:
                print(error_val, weights)
                return error_val, weights
        i += 1
    return error, weights


def converged(weights):
    has_converged = False
    sum = 0.0
    for i in weights:
        sum += i
    avg = sum/float(len(weights))
    if avg < 0.1:
        has_converged = True
    return has_converged, avg


def summed_gradient_plot(weights, step_size, data_points, classes):
    plot_petal_length_width(weights, lambda w0, w1, w2, x: -(w1 * x + w0) / w2, 'Original Decision Boundary')
    old_w = weights
    weights[0] -= update_w(old_w, step_size, data_points, classes, 0)
    weights[1] -= update_w(old_w, step_size, data_points, classes, 1)
    weights[2] -= update_w(old_w, step_size, data_points, classes, 2)
    plot_petal_length_width(weights,
                            lambda w0, w1, w2, x: -(w1 * x + w0) / w2, 'Decision Boundary after Summed Gradient')


def classify_prediction(val):
    # print(val)
    if val >= 0:
        return 1
    else:
        return -1


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
    x = range(0, 10)
    y = []
    for i in range(0, iterations):
        old_w = weights
        weights[0] -= update_w(old_w, step_size, data_points, classes, 0)
        weights[1] -= update_w(old_w, step_size, data_points, classes, 1)
        weights[2] -= update_w(old_w, step_size, data_points, classes, 2)
        # print('w0: ' + str(weights[0]) + ' | ' + 'w1: ' + str(weights[1]) + ' | ' + 'w2: ' + str(weights[2]))
        error = (mse(data_points, weights, classes))
        if i % 100 == 0:
            y.append(float(error))
    plt.plot(x, y)
    plt.show()


def plot_3c(iterations, weights, step_size, data_points, classes):
    x_error = range(0, 10)
    y_error = []
    initial_error = (mse(data_points, weights, classes))
    for i in range(0, iterations):
        old_w = weights
        weights[0] -= update_w(old_w, step_size, data_points, classes, 0)
        weights[1] -= update_w(old_w, step_size, data_points, classes, 1)
        weights[2] -= update_w(old_w, step_size, data_points, classes, 2)
        error = (mse(data_points, weights, classes))
        if i % 100 == 0:
            y_error.append(float(error))

    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    axes = plt.gca()
    axes.set_xlim([0, 10])
    axes.set_ylim([0, 1])

    plt.title('Learning curve: Initial')
    plt.plot(0, initial_error)
    plt.show()

    plt.title('Learning curve: Middle')
    axes = plt.gca()
    axes.set_xlim([0, 10])
    axes.set_ylim([0, 1])
    plt.plot(x_error[0:5], y_error[0:5])
    plt.show()

    plt.title('Learning curve: Final')
    axes = plt.gca()
    axes.set_xlim([0, 10])
    axes.set_ylim([0, 1])
    plt.plot(x_error[0:10], y_error[0:10])
    plt.show()


if __name__ == "__main__":
    data2 = read_csv()

    data, classes = get_data()
    # weights = [0, 10, 10]
    # weights = [-14.36, 1.562, 4.8]

    # plot_mse(data, weights, 0.001, classes)

    # 2B choose good and bad function and plot the boundary line over the data
    # mse = mse(data, [0, -0.95, -0.95], classes)
    # print(mse)
    # plot_weights = [-15.3, 1.4, 4.8] # good
    plot_weights = [-18.8, 1.786, 6.048]
    # plot_weights = [-15.36, 1.562, 4.8]

    # plot_petal_length_width(plot_weights,
    #                         lambda w0, w1, w2, x: -(w1 * x + w0) / w2, 'Hand Selected Decision Boundary')
    # print('(5.1, 1.8): ' + classify(5.1, 1.8, plot_weights, lambda x1, x2, w0, w1, w2: (w1 * x1 + w2 * x2) + w0)[0])
    # print('(4.8, 1.8): ' + classify(4.8, 1.8, plot_weights, lambda x1, x2, w0, w1, w2: (w1 * x1 + w2 * x2) + w0)[0])
    # print('(4.5, 1.5): ' + classify(4.5, 1.5, plot_weights, lambda x1, x2, w0, w1, w2: (w1 * x1 + w2 * x2) + w0)[0])
    # print('(3.3, 1): ' + classify(3.3, 1, plot_weights, lambda x1, x2, w0, w1, w2: (w1 * x1 + w2 * x2) + w0)[0])

    # plot_extra_credit([-0.6, 0.5, 0.5], [4.0, 1.0])
    # circle_classifier(4.7,1.4, [-0.6, 0.5, 0.5], [4.0, 1.0],
    #                   lambda x1, x2, w0, w1, w2, c0, c1: w1 * ((x1 - c0) ** 2) + w2 * ((x2 - c1) ** 2) + w0)
    # circle_classifier(4.7,1.4, [-0.6, 0.5, 0.5], [4.0, 1.0],
    #                   lambda x1, x2, w0, w1, w2, c0, c1: w1 * ((x1 - c0) ** 2) + w2 * ((x2 - c1) ** 2) + w0)
    # circle_classifier(5.1,1.9, [-0.6, 0.5, 0.5], [4.0, 1.0],
    #                   lambda x1, x2, w0, w1, w2, c0, c1: w1 * ((x1 - c0) ** 2) + w2 * ((x2 - c1) ** 2) + w0)
    # circle_classifier(5.7,2.3, [-0.6, 0.5, 0.5], [4.0, 1.0],
    #                   lambda x1, x2, w0, w1, w2, c0, c1: w1 * ((x1 - c0) ** 2) + w2 * ((x2 - c1) ** 2) + w0)

    # weights = [-18.8, 1.786, 6.048]
    # error = mse(data, weights, classes)
    # plot_petal_length_width(weights,
    #                         lambda w0, w1, w2, x: -(w1 * x + w0) / w2, 'Good Boundary | Error: ' + str(error))
    # weights = [-11, 1.5, 5]
    # error = mse(data, weights, classes)
    # plot_petal_length_width(weights,
    #                         lambda w0, w1, w2, x: -(w1 * x + w0) / w2, 'Bad Boundary | Error: ' + str(error))

    # weights = [-14, 1.5, 5]
    # summed_gradient_plot(weights, 0.001, data, classes)

    weights = [10, 12, 12]
    g_descent(5000, weights, 0.001, data, classes)
    # plot_mse(data, weights, 0.01, classes)

    # weights = [-1.528, 0.1562, 0.48677]
    # error = mse(data, weights, classes)
    # plot_petal_length_width(weights,
    #                         lambda w0, w1, w2, x: -(w1 * x + w0) / w2, 'Best Boundary | Error: ' + str(error))

    # -15.28, 1.56, 4.87
    # centers: 4.0, 1.0
    # circle weights: -0.6, 0.5, 0.5
    # lambda x1, x2, w0, w1, w2, c0, c1: w1 * ((x1 - c0) ** 2) + w2 * ((x2 - c1) ** 2) + w0

    # 1C: use decision boundary from part 1b to classify a few points
    # 1D: Extra credit... use the circle decision boudary to classify a few points
    # 2B: plot the mse for two decision boundaries (good and bad)
    # 2C: mathematical derivation of the gradient
    # 2D: Show gradient in scalar and vector form
    # 2E: Summed gradient... show plot of decision boudary before and after
    # 3C: Show learning curve and decision boundary for beginning, middle, and end
    # plot_3c(1000, weights, 0.001, data, classes)
