import numpy as np
import matplotlib as ml
import matplotlib.pyplot as plt
import random
import sys

data = np.loadtxt('assign1_data.txt')
#print data

x1 = data[:, 0]
print ("x1:", x1)

x2 = data[:, 1]
print ("x2:", x2)

z = data[:, 3]
print ("z:", z)

test_x1 = x1[75:]
test_x2 = x2[75:]
test_z = z[75:]

def plot_error_epoch(accuracy_arr, lr, batch_size, first_n):
    accuracy_arr_y = np.array(accuracy_arr)
    epochs = []
    for i in range(len(accuracy_arr)):
        epochs.append(i)
    epochs_x = np.array(epochs)
    plt.plot(epochs_x, accuracy_arr_y)
    title = ''
    title += 'lr:'
    title += str(lr)
    title += ' batch_size:'
    title += str(batch_size)
    title += ' first:'
    title += str(first_n)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('accurary')
    plt.show()

def plot_epochs_bar(epochs_bar):
    y_axis = epochs_bar
    x_axis = [5, 10, 25, 50, 75]
    ind = np.arange(len(x_axis))
    plt.bar(ind, y_axis)
    plt.xticks(ind, x_axis)
    plt.ylabel('epoch num when converge')
    plt.xlabel('training set size')
    plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def activation(x): # use sigmoid as activation function
    a = sigmoid(x)
    if a >= 0.5:
        return 1
    else:
        return 0

# create method to train and get result with different settings like online, minibatch, batch and learning rate
epochs_bar = []

def train_and_get_result(x1, x2, first_n, lr, batch_size):

    x1_first_n = x1[:first_n]
    x2_first_n = x2[:first_n]

    if batch_size > 75:
        batch_size = 75

    accuracy_arr = []

    w1 = random.random() * 3
    w2 = random.random() * 3
    b = random.random() * 3

    print "random initial w1:\t", w1
    print "random initial w2:\t", w2
    print "random initial b:\t", b

    epoch = 0
    while epoch <= 5000:

        # training
        i = 1
        for x1i, x2i, zi in zip(x1, x2, z):
            sum_error_w1 = 0
            sum_error_w2 = 0
            sum_error_b = 0
            error = zi - activation(x1i * w1 + x2i * w2 + b)
            #error = zi - x1i * w1 + x2i * w2 + b
            sum_error_w1 += error * x1i
            sum_error_w2 += error * x2i
            sum_error_b += error
            if i >= batch_size: # only update weights after one batch
                w1 = w1 + lr * sum_error_w1
                w2 = w2 + lr * sum_error_w2
                b = b + lr * sum_error_b
                sum_error_w1 = 0
                sum_error_w2 = 0
                sum_error_b = 0
                i = 1
                continue
            i += 1


        # get accuracy from the entire set for later plotting
        accuracy = 0.0
        for x1i, x2i, zi in zip(test_x1, test_x2, test_z):
            if zi == activation(x1i * w1 + x2i * w2 + b):
                accuracy += 1.0
        accuracy /= len(test_x1)
        if epoch >= 4 and epoch % 4 == 0:
            if epoch % 100 == 0: # print only every 100 epochs
                print "epoch:\t", epoch, "\tfirst:\t", first_n, "\t acc last 25:\t", accuracy

        accuracy_arr.append(accuracy)
        if accuracy >= 0.96:
            if first_n == 5:
                if len(epochs_bar) < 1:
                    epochs_bar.append(epoch)
            elif first_n == 10:
                if len(epochs_bar) < 2:
                    epochs_bar.append(epoch)
            elif first_n == 25:
                if len(epochs_bar) < 3:
                    epochs_bar.append(epoch)
            elif first_n == 50:
                if len(epochs_bar) < 4:
                    epochs_bar.append(epoch)
            elif first_n == 75:
                if len(epochs_bar) < 5:
                    epochs_bar.append(epoch)
        epoch += 1

    plot_error_epoch(accuracy_arr, lr, batch_size, first_n)
    print "Result: w1:\t", w1, "\tw2:\t", w2, "\tb:\t", b, "\tnum of epoch:\t", epoch

# online
# learning rate lr = 0.0002
train_and_get_result(x1, x2, first_n = 5, lr = 0.0002, batch_size = 1)
train_and_get_result(x1, x2, first_n = 10, lr = 0.0002, batch_size = 1)
train_and_get_result(x1, x2, first_n = 25, lr = 0.0002, batch_size = 1)
train_and_get_result(x1, x2, first_n = 50, lr = 0.0002, batch_size = 1)
train_and_get_result(x1, x2, first_n = 75, lr = 0.0002, batch_size = 1)

plot_epochs_bar(epochs_bar)

# reference: http://cs229.stanford.edu/notes/cs229-notes1.pdf
