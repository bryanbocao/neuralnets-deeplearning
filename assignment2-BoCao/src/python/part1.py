
# coding: utf-8

# In[39]:


'''
Author: Bryan Bo Cao
Email: boca7588@colorado.edu or bo.cao-1@colorado.edu
Github Repo: https://github.com/BryanBo-Cao/neuralnets-deeplearning

Note that input layer is noted as layer0, hidden layer is noted as layer1, and output layer is noted is layer2
'''
import numpy as np
import random
import copy
import matplotlib.pyplot as plt

train_data = np.genfromtxt('../data/assign2_train_data.txt', delimiter=',')
#print "train_data:", train_data

t = train_data[:,2]
hu = train_data[:,3]
lt = train_data[:,4]
co2 = train_data[:,5]
hu_r = train_data[:,6]

o = train_data[:,7]

#data
data = np.column_stack((t, hu, lt, co2, hu_r))
print "data:", data

print "np.shape(train_data): ", np.shape(train_data)
#print "np.shape(t): ", np.shape(t)

print "t: ", t
print "hu: ", hu
print "lt: ", lt
print "co2: ", co2
print "hu_r: ", hu_r
print "o: ", o


# In[40]:


def normalize(data):
    #normalize train data
    #normalize t
    t = data[:, 0]
    t_min = np.min(t)
    t_max = np.max(t)
    d_t = t_max - t_min
    #normalized t: n_t
    n_t = []
    for each in t:
        n_t.append((each - t_min) / d_t)

    #normalize hu
    hu = data[:, 1]
    hu_min = np.min(hu)
    hu_max = np.max(hu)
    d_hu = hu_max - hu_min
    #normalized h: n_hu
    n_hu = []
    for each in hu:
        n_hu.append((each - hu_min) / d_hu)

    #normalize lt
    lt = data[:, 2]
    lt_min = np.min(lt)
    lt_max = np.max(lt)
    d_lt = lt_max - lt_min
    #normalized lt: n_lt
    n_lt = []
    for each in lt:
        n_lt.append((each - lt_min) / d_lt)

    #normalize co2
    co2 = data[:, 3]
    co2_min = np.min(co2)
    co2_max = np.max(co2)
    d_co2 = co2_max - co2_min
    #normalized co2: n_co2
    n_co2 = []
    for each in co2:
        n_co2.append((each - co2_min) / d_co2)

    #normalize hu_r
    hu_r = data[:, 4]
    hu_r_min = np.min(hu_r)
    hu_r_max = np.max(hu_r)
    d_hu_r = hu_r_max - hu_r_min
    #normalized hu_r: n_hu_r
    n_hu_r = []
    for each in hu_r:
        n_hu_r.append((each - hu_r_min) / d_hu_r)

    #normalized data: n_data
    n_data = np.column_stack((n_t, n_hu, n_lt, n_co2, n_hu_r))

    return n_data


# In[41]:


n_data = normalize(data)
print "n_data:", n_data


# In[46]:


def initialize_weights():
    w_t = random.random() * 2
    w_hu = random.random() * 2
    w_lt = random.random() * 2
    w_co2 = random.random() * 2
    w_hu_r = random.random() * 2
    b = random.random() * 2

    #weights
    ws = [w_t, w_hu, w_lt, w_co2, w_hu_r, b]
    return ws


init_ws = initialize_weights()
ws = copy.deepcopy(init_ws)
print "ws: ", ws


# In[47]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def activation(x): # use sigmoid as activation function
    a = sigmoid(x)
    if a >= 0.5:
        return 1
    else:
        return 0

def perceptron(ins, ws):
    sum = 0
    for x, w in zip(ins, ws):
        sum += x * w
    return activation(sum)

# lr - learning rate
# bs - batch size

def train(n_data, lr, bs):
    test_data = np.genfromtxt('../data/assign2_test_data.txt', delimiter=',')
    o_test = test_data[:, 7]
    n_test_data = get_normalized_test_data(test_data)

    print "Training starts: len of data:", len(n_data)

    train_accuracies = []
    test_accuracies = []
    epoch = 0
    n_t = n_data[:, 0]
    n_hu = n_data[:, 1]
    n_lt = n_data[:, 2]
    n_co2 = n_data[:, 3]
    n_hu_r = n_data[:, 4]
    error = 0
    while epoch <= 1000:
        # train
        i = 0
        for (n_t_i, n_hu_i, n_lt_i, n_co2_i, n_hu_r_i, o_i) in zip(n_t, n_hu, n_lt, n_co2, n_hu_r, o):
            sum_error_w_t = 0
            sum_error_w_hu = 0
            sum_error_w_lt = 0
            sum_error_w_co2 = 0
            sum_error_w_hu_r = 0
            sum_error_b = 0

            #error = o_i - (n_t_i * ws[0] + n_hu_i * ws[1] + n_lt_i * ws[2] + n_co2_i * ws[3] + n_hu_r_i * ws[4] + ws[5])
            error = o_i - sigmoid(n_t_i * ws[0] + n_hu_i * ws[1] + n_lt_i * ws[2] + n_co2_i * ws[3] + n_hu_r_i * ws[4] + ws[5])
            #print "error: ", error
            sum_error_w_t += error * n_t_i
            sum_error_w_hu += error * n_hu_i
            sum_error_w_lt += error * n_lt_i
            sum_error_w_co2 += error * n_co2_i
            sum_error_w_hu_r += error * n_hu_r_i
            sum_error_b += error

            if i >= bs: # only update weights after one batch size
                ws[0] += lr * sum_error_w_t
                ws[1] += lr * sum_error_w_hu
                ws[2] += lr * sum_error_w_lt
                ws[3] += lr * sum_error_w_co2
                ws[4] += lr * sum_error_w_hu_r
                ws[5] += lr * sum_error_b
                sum_error_w_t = 0
                sum_error_w_hu = 0
                sum_error_w_lt = 0
                sum_error_w_co2 = 0
                sum_error_w_hu_r = 0
                sum_error_b = 0
                i = 1
                continue

            #test every epoch
            train_accuracy = get_one_feedforward_accuracy(n_data, ws, o)
            train_accuracies.append(train_accuracy)
            test_accuracy = test(n_test_data, ws, o_test)
            test_accuracies.append(test_accuracy)
            if epoch % 10 == 0:
                print "   "
                print "epoch:", epoch
                print "train accuracy:\t", train_accuracy
                print "Test accuracy:\t", test_accuracy
                print "ws :", ws
                print "error: ", error
            i += 1

        # count
        epoch += 1
    print "Training ends."
    print "Initial weights:", init_ws
    print "Trained weights:", ws
    print "Trained data accuracy:", train_accuracies[-1]

    # print "test_data:", test_data
    # test(test_data, ws, bs = 1)

    plot_accuracy(train_accuracies, test_accuracies, lr, bs)

def get_normalized_test_data(test_data):
    test_t = test_data[:,2]
    test_hu = test_data[:,3]
    test_lt = test_data[:,4]
    test_co2 = test_data[:,5]
    test_hu_r = test_data[:,6]

    new_test_data = np.column_stack((test_t, test_hu, test_lt, test_co2, test_hu_r))
    # print "new_test_data:", new_test_data

    #normalize test data
    return normalize(new_test_data)

def test(n_test_data, ws, o_test):
    # print "Testing starts: len of data:", len(test_data)
    test_accuracy = get_one_feedforward_accuracy(n_test_data, ws, o_test)
    return test_accuracy

def get_one_feedforward_accuracy(n_data_t, ws_t, o_t):

    #calculate accuracy
    #outputs
    os_t = []
    for each in zip(n_data_t):
        for ins_t in each: # each sample
            ins_t = ins_t.tolist()
            ins_t.append(1)
            sum_ = np.inner(ins_t, ws_t)
            os_t.append(activation(sum_))

    #count correct output number
    correct_cnt = 0
    for o_t_i, os_t_i in zip(o_t, os_t):
        if (o_t_i == os_t_i):
            correct_cnt += 1
    accuracy = float(correct_cnt) / float(len(o_t))
    ##calculate accuracy
    return accuracy

def plot_accuracy(train_accuracies, test_accuracies, lr, bs):
    train_accs_y = np.array(train_accuracies)
    epochs = []
    for i in range(len(train_accuracies)):
        epochs.append(i)
    epochs_x = np.array(epochs)
    plt.plot(epochs_x, train_accs_y, label = 'train accuracy')
    test_accs_y = np.array(test_accuracies)
    plt.plot(epochs_x, test_accs_y, label = 'test accuracy')
    title = ' lr:'
    title += str(lr)
    title += ' batch_size:'
    title += str(bs)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


# In[48]:


train(n_data, lr = 0.0001, bs = 1)
