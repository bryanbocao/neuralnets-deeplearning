'''
Author: Bryan Bo Cao
Email: boca7588@colorado.edu or bo.cao-1@colorado.edu
Github Repo: https://github.com/BryanBo-Cao/neuralnets-deeplearning

Note that input layer is noted as layer0, hidden layer is noted as layer1, and output layer is noted is layer2
'''
import numpy as np
import matplotlib.pyplot as plt

H = [1, 2, 5, 10, 20]
train_accuracies = [0.787670391748, 0.984772196979, 0.985017806705, 0.976789880879, 0.98600024561]
test_accuracies = [0.789889253486, 0.988310090238, 0.992616899098, 0.992821985234, 0.991899097621]

def plot_performance_bar(H, train_accuracies, test_accuracies):
    y_axis_0 = train_accuracies
    y_axis_1 = test_accuracies
    x_axis = H
    ind = np.arange(len(x_axis))
    plt.bar(ind - 0.2, y_axis_0, width = 0.2, color = 'b', align = 'center')
    plt.bar(ind, y_axis_1, width = 0.2, color = 'r', align = 'center')
    plt.xticks(ind, x_axis)
    plt.ylabel('Accuracy')
    plt.legend( (y_axis_0, y_axis_1), ('Train Accuracy', 'Test Accuracy') )
    plt.xlabel('Number of hidden layer units')
    plt.show()

plot_performance_bar(H, train_accuracies, test_accuracies)
