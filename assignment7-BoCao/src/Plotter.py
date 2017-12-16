"""
Author: Bryan Bo Cao
Email: boca7588@colorado.edu or bo.cao-1@colorado.edu
Github Repo: https://github.com/BryanBo-Cao/neuralnets-deeplearning
Reference:
    https://github.com/tylersco/deep-neural-networks-art-images
    http://www.scipy-lectures.org/advanced/image_processing/
"""
import matplotlib.pyplot as plt

def plot_2lists(x, y, x_label, y_label):
    plt.figure(figsize=(10, 6))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x, y, 'ro')
    plt.show()
    
def plot_4lists(alpha_x, alpha_y, beta_x, beta_y, x_label, y_label):
    plt.figure(figsize=(10, 6))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(alpha_x, alpha_y, 'ro', beta_x, beta_y, 'bo')
    plt.show()

def plot_acc_distance(data):
    plt.figure(figsize=(10, 6))
    plt.xlabel("Distance")
    plt.ylabel("Human Judgement Accuracy")
    
    alpha_x = []
    beta_x = []
    alpha_x.extend(data.alpha_pairs['same'][3])
    alpha_x.extend(data.alpha_pairs['diff'][3])
    beta_x.extend(data.beta_pairs['same'][3])
    beta_x.extend(data.beta_pairs['diff'][3])
    #print ("x,", x)
    
    alpha_y = []
    beta_y = []
    alpha_y.extend(data.alpha_pairs['same'][4])
    alpha_y.extend(data.alpha_pairs['diff'][4])
    beta_y.extend(data.beta_pairs['same'][4])
    beta_y.extend(data.beta_pairs['diff'][4])
    #print("y, ", y)
    
    plt.plot(alpha_x, alpha_y, 'ro', beta_x, beta_y, 'bo')
    plt.ylim(0, 1)
    plt.show()
    
def plot_acc_cosine(data):
    plt.figure(figsize=(10, 6))
    plt.xlabel("Cosine")
    plt.ylabel("Human Judgement Accuracy")
    
    alpha_x = []
    beta_x = []
    alpha_x.extend(data.alpha_pairs['same'][2])
    alpha_x.extend(data.alpha_pairs['diff'][2])
    beta_x.extend(data.beta_pairs['same'][2])
    beta_x.extend(data.beta_pairs['diff'][2])
    #print ("x,", x)
    
    alpha_y = []
    beta_y = []
    alpha_y.extend(data.alpha_pairs['same'][4])
    alpha_y.extend(data.alpha_pairs['diff'][4])
    beta_y.extend(data.beta_pairs['same'][4])
    beta_y.extend(data.beta_pairs['diff'][4])
    #print("y, ", y)
    
    plt.plot(alpha_x, alpha_y, 'ro', beta_x, beta_y, 'bo')
    plt.ylim(0, 1)
    plt.show()
    
def plot_prob_distance(data):
    plt.figure(figsize=(10, 6))
    plt.xlabel("Distance")
    plt.ylabel("Probability of human judgement on similarity")
    
    alpha_x = []
    beta_x = []
    alpha_x.extend(data.alpha_pairs['same'][3])
    alpha_x.extend(data.alpha_pairs['diff'][3])
    beta_x.extend(data.beta_pairs['same'][3])
    beta_x.extend(data.beta_pairs['diff'][3])
    #print ("x,", x)
    
    alpha_y = []
    beta_y = []
    alpha_y.extend(data.alpha_pairs['same'][5])
    alpha_y.extend(data.alpha_pairs['diff'][5])
    beta_y.extend(data.beta_pairs['same'][5])
    beta_y.extend(data.beta_pairs['diff'][5])
    #print("y, ", y)
    
    plt.plot(alpha_x, alpha_y, 'ro', beta_x, beta_y, 'bo')
    plt.ylim(0, 1)
    plt.show()
    
def plot_prob_cosine(data):
    plt.figure(figsize=(10, 6))
    plt.xlabel("Cosine")
    plt.ylabel("Probability of human judgement on similarity")
    
    alpha_x = []
    beta_x = []
    alpha_x.extend(data.alpha_pairs['same'][2])
    alpha_x.extend(data.alpha_pairs['diff'][2])
    beta_x.extend(data.beta_pairs['same'][2])
    beta_x.extend(data.beta_pairs['diff'][2])
    #print ("x,", x)
    
    alpha_y = []
    beta_y = []
    alpha_y.extend(data.alpha_pairs['same'][5])
    alpha_y.extend(data.alpha_pairs['diff'][5])
    beta_y.extend(data.beta_pairs['same'][5])
    beta_y.extend(data.beta_pairs['diff'][5])
    #print("y, ", y)
    
    plt.plot(alpha_x, alpha_y, 'ro', beta_x, beta_y, 'bo')
    plt.ylim(0, 1)
    plt.show()