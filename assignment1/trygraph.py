
# coding: utf-8

# In[ ]:


import numpy as np
import pylab as plt

y1 = (10, 17, 9, 12, 3)
y2 = (22, 10, 15, 25, 13)
y3 = tuple(reversed(y1))  # generated for explanation
y4 = tuple(reversed(y2))  # generated for explanation
mydata = [y1, y2, y3, y4]

# plot graphs
for count, y_data in enumerate(mydata):
    x_data = range(1, len(y_data) + 1)
    print x_data
    print y_data
    plt.subplot(2, 2, count + 1)
    plt.plot(x_data, y_data, '-*')
    plt.xlabel('Trials')
    plt.ylabel('Frequency')
    plt.grid(True)

plt.show()

