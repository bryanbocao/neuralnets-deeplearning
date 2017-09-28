
# coding: utf-8

# In[3]:


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

cnt = 0
for ii in range(len(o)):
    if o[ii] == 1:
        cnt += 1
print "Train accuracy with all outputs 0:\t", (float(cnt) / float(len(o)))


test_data = np.genfromtxt('../data/assign2_test_data.txt', delimiter=',')

to = test_data[:,7]

cnt = 0
for ii in range(len(to)):
    if to[ii] == 1:
        cnt += 1
print "Test accuracy with all outputs 0:\t", (float(cnt) / float(len(to)))
