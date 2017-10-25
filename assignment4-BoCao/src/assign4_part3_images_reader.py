
# coding: utf-8

# In[1]:


'''
Author: Bryan Bo Cao
Email: boca7588@colorado.edu or bo.cao-1@colorado.edu
Github Repo: https://github.com/BryanBo-Cao/x
Github Repo: https://github.com/BryanBo-Cao/neuralnets-deeplearning (will be published after Dec 15, 2017)

This dataset is collected manually to test the out-of-class of the 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck are labelled with 0 - 9 respectively.
Out-of-class examples are labelled with all zeros.

In this dataset, 33 of 50 images are out of class.

To show image:
imgplot = plt.imshow(image)
plt.show()
'''
import numpy as np
from PIL import Image
import matplotlib.image as img
import matplotlib.pyplot as plt


# In[2]:


def get_data():
    image_data = np.zeros((50,32,32,3))
    for i in range(50):
        filename = "assign4_part3_data/" + str(i) + ".jpg"
        image = img.imread(filename)
        image_data[i] = image
    return image_data

def show_image(image_data, i):
    imgplot = plt.imshow(np.uint8(image_data[i]))
    plt.show()


# In[3]:


#image_data = get_data()
#print ("image_data.shape: ", image_data.shape)

#show_image(image_data, 3)
#show_image(image_data, 10)


# In[4]:


def get_labels():
    labels = np.zeros((50, 10))
    labels[0] = [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]
    labels[1] = [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]
    labels[2] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    labels[3] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    labels[4] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    labels[5] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    labels[6] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    labels[7] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    labels[8] = [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]
    labels[9] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    
    labels[10] = [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]
    labels[11] = [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]
    labels[12] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    labels[13] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    labels[14] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    labels[15] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    labels[16] = [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]
    labels[17] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    labels[18] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    labels[19] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    
    labels[20] = [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]
    labels[21] = [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]
    labels[22] = [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]
    labels[23] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    labels[24] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    labels[25] = [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]
    labels[26] = [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]
    labels[27] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]
    labels[28] = [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    labels[29] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    
    labels[30] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    labels[31] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    labels[32] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    labels[33] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    labels[34] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    labels[35] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    labels[36] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    labels[37] = [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]
    labels[38] = [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]
    labels[39] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    
    labels[40] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    labels[41] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    labels[42] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    labels[43] = [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]
    labels[44] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    labels[45] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    labels[46] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    labels[47] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    labels[48] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    labels[49] = [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]
    
    return labels


# In[5]:


#labels = get_labels()
#print ("labels.shape: ", labels.shape)

