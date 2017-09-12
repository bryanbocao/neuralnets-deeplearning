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

def plot_error_epoch(accuracy_arr, lr, batch_size):
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
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('accurary')
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
def train_and_get_result(x1, x2, lr, batch_size):

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
    while epoch <= 6000:

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

        if epoch >= 4 and epoch % 4 == 0:
            if epoch % 100 == 0: # print only every 100 epochs
                print "epoch:\t", epoch, "\t acc entire set:\t", accuracy
                #print "epoch:\t", epoch, "\t curr_validation_acc:\t", curr_validation_acc, "\tacc entire set:\t", accuracy

        # get accuracy from the entire set for later plotting
        accuracy = 0.0
        for x1i, x2i, zi in zip(x1, x2, z):
            if zi == activation(x1i * w1 + x2i * w2 + b):
                accuracy += 1.0
        accuracy /= len(x1)

        accuracy_arr.append(accuracy)
        epoch += 1

    plot_error_epoch(accuracy_arr, lr, batch_size)
    print "Result: w1:\t", w1, "\tw2:\t", w2, "\tb:\t", b, "\tnum of epoch:\t", epoch

# online
# learning rate lr = 0.0001
train_and_get_result(x1, x2, lr = 0.0001, batch_size = 1)

# reference: http://cs229.stanford.edu/notes/cs229-notes1.pdf
