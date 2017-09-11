Student: Bo Cao
Email: bo.cao-1@colorado.edu
Each python code is written in *.ipynb file.
Please use Jupyter to open these files.

Part1:
(1a) Report the values of w1, w2, and b.
Me: 
w1: -2.04424259514
w2: 3.99686016866
b: -0.924290811868
(1b) What function or method did you use to find the least-squares solution?
Me: Instead of using some functions directly from some packages, I wrote the code for my own. First calculate the mean of each vector x1, x2 and y, then w1 = sum((x1 - x1_mean) * (y - y_mean)) / sum(square(x1 - x1_mean)), w2 = sum((x2 - x2_mean) * (y - y_mean)) / sum(square(x2 - x2_mean)), b = y_mean - w1 * x1_mean - w2 * x2_mean.
Reference: https://jonathantemplin.com/files/regression/ersh8320f07/ersh8320f07_06.pdf
http://faculty.cas.usf.edu/mbrannick/regression/Reg2IV.html

Part2:
To update weight: wj = wj + lr * sum((yi - yi_predicted) * xji)
(2a) Report the values of w1, w2, and b.
(2b) What settings worked well for you:  online vs. batch vs. minibatch? what step size? how did you decide to terminate?
(2c) Make a graph of error on the entire data set as a function of epoch. An epoch is a complete sweep through all the data.

Reference:
LMS algorithm: http://cs229.stanford.edu/notes/cs229-notes1.pdf
Initialise learning rate: http://web.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf
Early Stop: http://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf


