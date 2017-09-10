Each python code is written in *.ipynb file.
Please use Jupyter to open these files.

Part1:
(1a) Report the values of w1, w2, and b.
Me: w1 = -1.50040034722, w2 = 3.71016742759, b = -1.02907821007
(1b) What function or method did you use to find the least-squares solution?
Instead of using some functions directly from some packages, I wrote the code for my own. First calculate the mean of each vector x1, x2 and y, then w1 = sum((x1 - x1_mean) * (y - y_mean)) / sum(square(x1 - x1_mean)), w2 = sum((x2 - x2_mean) * (y - y_mean)) / sum(square(x2 - x2_mean)), b = y_mean - w1 * x1_mean - w2 * x2_mean.
