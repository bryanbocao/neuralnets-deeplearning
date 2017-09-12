import numpy as np

data = np.loadtxt('assign1_data.txt')
#print data

x1 = data[:, 0]
print "x1:", x1

x2 = data[:, 1]
print "x2:", x2

y = data[:, 2]
#print "y:", y

x1_mean = np.mean(x1)
x2_mean = np.mean(x2)
y_mean = np.mean(y)

# reference: https://jonathantemplin.com/files/regression/ersh8320f07/ersh8320f07_06.pdf
sum_y = sum(y)
sum_x1 = sum(x1)
sum_x2 = sum(x2)
sum_y_2 = sum(y ** 2)

N = float(len(y))
sum_mini_y_2 = sum(y ** 2) - sum(y) ** 2 / N
sum_mini_x1_2 = sum(x1 ** 2) - sum(x1) ** 2 / N
sum_mini_x2_2 = sum(x2 ** 2) - sum(x2) ** 2 / N
sum_mini_x1_y = sum(x1 * y) - sum(x1) * sum(y) / N
sum_mini_x2_y = sum(x2 * y) - sum(x2) * sum(y) / N
sum_mini_x1_x2 = sum(x1 * x2) - sum(x1) * sum(x2) / N

w1 = (sum_mini_x2_2 * sum_mini_x1_y - sum_mini_x1_x2 * sum_mini_x2_y) / (sum_mini_x1_2 * sum_mini_x2_2 - sum_mini_x1_x2 ** 2)
w2 = (sum_mini_x1_2 * sum_mini_x2_y - sum_mini_x1_x2 * sum_mini_x1_y) / (sum_mini_x2_2 * sum_mini_x1_2 - sum_mini_x1_x2 ** 2)
b = y_mean - w1 * x1_mean - w2 * x2_mean

print "w1:\t", w1
print "w2:\t", w2
print "b:\t", b
