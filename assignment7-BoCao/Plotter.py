"""
Author: Bryan Bo Cao
Email: boca7588@colorado.edu or bo.cao-1@colorado.edu
Github Repo: https://github.com/BryanBo-Cao/neuralnets-deeplearning
Reference:
    https://github.com/tylersco/deep-neural-networks-art-images
    http://www.scipy-lectures.org/advanced/image_processing/
"""
import matplotlib.pyplot as plt

def plot_acc_distance(data):
    plt.xlabel("distance")
    plt.ylabel("human judgement accuracy")
    
    x = []
    x.extend(data.alpha_pairs['same'][3])
    x.extend(data.beta_pairs['same'][3])
    x.extend(data.alpha_pairs['diff'][3])
    x.extend(data.beta_pairs['diff'][3])
    #print ("x,", x)
    
    y = []
    y.extend(data.alpha_pairs['same'][4])
    y.extend(data.beta_pairs['same'][4])
    y.extend(data.alpha_pairs['diff'][4])
    y.extend(data.beta_pairs['diff'][4])
    #print("y, ", y)
    
    plt.plot(x, y, 'ro')
    plt.ylim(0, 1)
    plt.show()
    
def plot_acc_cosine(data):
    plt.xlabel("cosine")
    plt.ylabel("human judgement accuracy")
    
    x = []
    x.extend(data.alpha_pairs['same'][2])
    x.extend(data.beta_pairs['same'][2])
    x.extend(data.alpha_pairs['diff'][2])
    x.extend(data.beta_pairs['diff'][2])
    #print ("x,", x)
    
    y = []
    y.extend(data.alpha_pairs['same'][4])
    y.extend(data.beta_pairs['same'][4])
    y.extend(data.alpha_pairs['diff'][4])
    y.extend(data.beta_pairs['diff'][4])
    #print("y, ", y)
    
    plt.plot(x, y, 'ro')
    #plt.xlim(0, 1)
    #plt.ylim(0, 1)
    plt.show()