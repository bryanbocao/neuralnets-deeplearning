
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

y = data[:, 2]
#print ("y:", y)
'''
solution from part1 is:
w1:	-2.04424259514
w2:	3.99686016866
b:	-0.924290811868
'''


# In[18]:


def plot_error_epoch(errors, lr, batch_size):
    errors_y = np.array(errors)
    epochs = []
    for i in range(len(errors)):
        epochs.append(i)
    epochs_x = np.array(epochs)
    plt.plot(epochs_x, errors_y)
    title = ''
    title += 'lr:'
    title += str(lr)
    title += ' batch_size:'
    title += str(batch_size)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.show()


# In[16]:


# create method to train and get result with different settings like online, minibatch, batch and learning rate
def train_and_get_result(x1, x2, lr, batch_size):

    if batch_size > 75:
        batch_size = 75

    errors = []

    w1 = random.random() * 3
    w2 = random.random() * 3
    b = random.random() * 3

    print "random initial w1:\t", w1
    print "random initial w2:\t", w2
    print "random initial b:\t", b

    epoch = 0
    # training set size : validation set size = 3 : 1
    training_set_index = 0
    size = len(x1)
    validation_set_index = 0
    validation_set_index_step = size / 4
    pre_validation_error = sys.maxint
    curr_validation_error = sys.maxint - 1
    while curr_validation_error >= 1 and (pre_validation_error >= curr_validation_error or curr_validation_error <= 0): # early stopping when the validation error is greater than its previous one
    #while True:

        validation_set_index += validation_set_index_step
        validation_set_index %= size

        x1_validation = x1[validation_set_index : validation_set_index + validation_set_index_step]
        x2_validation = x2[validation_set_index : validation_set_index + validation_set_index_step]
        y_validation = y[validation_set_index : validation_set_index + validation_set_index_step]

        # training
        i = 1
        for x1i, x2i, yi in zip(x1, x2, y):
            sum_error_w1 = 0
            sum_error_w2 = 0
            sum_error_b = 0
            if x1i not in zip(x1_validation) and x2i not in zip(x2_validation) and yi not in zip(y_validation):
                error = yi - (x1i * w1 + x2i * w2 + b)
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
            pre_validation_error = curr_validation_error
            if epoch % 100 == 0: # print only every 20 epochs
                print "epoch:\t", epoch, "\t curr_validation_error:\t", curr_validation_error
            curr_validation_error = 0

        # validation
        for x1i, x2i, yi in zip(x1_validation, x2_validation, y_validation):
            curr_validation_error += abs(yi - (x1i * w1 + x2i * w2 + b))

        # get error from the entire set for later plotting
        error = 0
        for x1i, x2i, yi in zip(x1, x2, y):
            error += yi - (x1i * w1 + x2i * w2 + b)
        errors.append(abs(error))
        epoch += 1

    plot_error_epoch(errors, lr, batch_size)
    print "Result: pre_validation_error:\t", pre_validation_error, "curr_validation_error:\t", curr_validation_error
    print "Result: w1:\t", w1, "\tw2:\t", w2, "\tb:\t", b, "\tnum of epoch:\t", epoch


# In[17]:


# online
# learning rate lr = 0.0001
train_and_get_result(x1, x2, lr = 0.001, batch_size = 1)


# In[ ]:


# online
# learning rate lr = 0.0005
train_and_get_result(x1, x2, lr = 0.0005, batch_size = 1)


# In[11]:


# online
# learning rate lr = 0.001
train_and_get_result(x1, x2, lr = 0.001, batch_size = 1)


# In[16]:


# minibatch batch_size = 50
# learning rate lr = 0.0001
train_and_get_result(x1, x2, lr = 1, batch_size = 1)


# In[6]:


# online
# learning rate lr = 0.005
train_and_get_result(x1, x2, lr = 0.005, batch_size = 1)


# In[ ]:


# online
# learning rate lr = 0.01
train_and_get_result(x1, x2, lr = 1, batch_size = 1)


# In[8]:


# minibatch batch_size = 5
# learning rate lr = 0.0001
train_and_get_result(x1, x2, lr = 0.0001, batch_size = 5)


# In[9]:


# minibatch batch_size = 25
# learning rate lr = 0.0001
train_and_get_result(x1, x2, lr = 0.0001, batch_size = 25)


# In[ ]:


# minibatch batch_size = 50
# learning rate lr = 0.0001
train_and_get_result(x1, x2, lr = 0.0001, batch_size = 50)


# In[ ]:





# In[ ]:


# minibatch batch_size = 5
# learning rate lr = 0.0005
train_and_get_result(x1, x2, lr = 0.0005, batch_size = 5)


# In[ ]:


# minibatch batch_size = 25
# learning rate lr = 0.0005
train_and_get_result(x1, x2, lr = 0.0005, batch_size = 25)


# In[ ]:


# minibatch batch_size = 50
# learning rate lr = 0.0005
train_and_get_result(x1, x2, lr = 0.0005, batch_size = 50)


# In[ ]:


# minibatch batch_size = 5
# learning rate lr = 0.001
train_and_get_result(x1, x2, lr = 0.001, batch_size = 5)


# In[ ]:


# minibatch batch_size = 25
# learning rate lr = 0.001
train_and_get_result(x1, x2, lr = 0.001, batch_size = 25)


# In[ ]:


# minibatch batch_size = 50
# learning rate lr = 0.001
train_and_get_result(x1, x2, lr = 0.001, batch_size = 50)


# In[ ]:


# minibatch batch_size = 5
# learning rate lr = 0.005
train_and_get_result(x1, x2, lr = 0.005, batch_size = 5)


# In[ ]:


# minibatch batch_size = 25
# learning rate lr = 0.005
train_and_get_result(x1, x2, lr = 0.005, batch_size = 25)


# In[ ]:


# minibatch batch_size = 50
# learning rate lr = 0.005
train_and_get_result(x1, x2, lr = 0.005, batch_size = 50)


# In[ ]:


# batch
# learning rate lr = 0.0001
train_and_get_result(x1, x2, lr = 0.0001, batch_size = 100)


# In[ ]:


# batch
# learning rate lr = 0.0005
train_and_get_result(x1, x2, lr = 0.0005, batch_size = 100)


# In[ ]:


# batch
# learning rate lr = 0.001
train_and_get_result(x1, x2, lr = 0.001, batch_size = 100)


# In[ ]:


# batch
# learning rate lr = 0.005
train_and_get_result(x1, x2, lr = 0.005, batch_size = 100)


# reference: http://cs229.stanford.edu/notes/cs229-notes1.pdf
