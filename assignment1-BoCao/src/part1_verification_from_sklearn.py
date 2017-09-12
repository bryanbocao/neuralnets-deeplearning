import numpy as np
from sklearn.linear_model import LinearRegression

data = np.loadtxt('assign1_data.txt')
#print data

x1 = data[:, 0]
#print "x1:", x1

x2 = data[:, 1]
#print "x2:", x2

y = data[:, 2]
#print "y:", y

x = np.array((x1, x2)).T
l = LinearRegression()
l.fit(x, y)

w1 = l.coef_[0]
w2 = l.coef_[1]
b = l.intercept_
print "w1:\t", w1
print "w2:\t", w2
print "b:\t", b
