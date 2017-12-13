"""
Author: Bryan Bo Cao
Email: boca7588@colorado.edu or bo.cao-1@colorado.edu
Github Repo: https://github.com/BryanBo-Cao/neuralnets-deeplearning
Reference:
    https://github.com/tylersco/deep-neural-networks-art-images
    http://www.scipy-lectures.org/advanced/image_processing/
"""
import numpy as np
from scipy import misc
import os, sys
import glob
import pandas as pd
import csv

class ArtData:
    '''
    ALPHA
        48 cubist-cubist
        48 impressionist-impressionist
        96 cubist vs impressionist

    BETA
        48 cubist-cubist
        48 impressionist-impressionist
        96 cubist vs impressionist

    Images
        1296 * 2 = 2592
    '''

    def __init__(self):
        
        self.metadata_folder = os.getcwd() + '/data/'

        self.train_images = {}
        self.alpha_pairs = {
            'same': [],
            'diff': [],
        }
        self.beta_pairs = {
            'same': [],
            'diff': []
        }
        print("ArtData Initialized!")

    def load_images(self):
        files = glob.glob(self.metadata_folder + 'img/*.jpg')
        for file in files:
            image = misc.imread(file)
            image = np.array(image) / 255.0
            image_name_index = file.rfind('/') + 1
            image_name = file[image_name_index:]
            self.train_images[image_name] = image
            #print(image_name)
        if (len(self.train_images) > 0):
            print("All images loaded!")
        
    def load_metadata(self):
        df_c_c = pd.read_csv(self.metadata_folder + "metadata-c-c.csv",
                         names=['Cubist-Cubist-A', 'Cubist-Cubist-B', 'Cubist-Cubist-Cosine',
                                'Same/Diff', 'N', 'Percent-Correct'])

        df_i_i = pd.read_csv(self.metadata_folder + "metadata-i-i.csv",
                         names=['Impressionist-Impressionist-A', 'Impressionist-Impressionist-B', 'Impressionist-Impressionist-Cosine',
                                'Same/Diff', 'N', 'Percent-Correct'])

        df_c_i = pd.read_csv(self.metadata_folder + "metadata-c-i.csv",
                         names=['Cubist-Impressionist-A', 'Cubist-Impressionist-B', 'Cubist-Impressionist-Cosine',
                                'Same/Diff', 'N', 'Percent-Correct'])

        # alpha same
        a_c_c = df_c_c['Cubist-Cubist-A'][1:49]
        b_c_c = df_c_c['Cubist-Cubist-B'][1:49]
        consine_c_c = df_c_c['Cubist-Cubist-Cosine'][1:49]
        accuracy_c_c = df_c_c['Percent-Correct'][1:49]

        a_i_i = df_i_i['Impressionist-Impressionist-A'][1:49]
        b_i_i = df_i_i['Impressionist-Impressionist-B'][1:49]
        consine_i_i = df_i_i['Impressionist-Impressionist-Cosine'][1:49]
        accuracy_i_i = df_i_i['Percent-Correct'][1:49]

        a = []
        a.extend(a_c_c)
        a.extend(a_i_i)

        b = []
        b.extend(b_c_c)
        b.extend(b_i_i)

        consine = []
        consine.extend(consine_c_c)
        consine.extend(consine_i_i)

        accuracy = []
        accuracy.extend(accuracy_c_c)
        accuracy.extend(accuracy_i_i)
        self.alpha_pairs['same'] = [a, b, consine, accuracy]
        #print (self.alpha_pairs['same'])

        # alpha diff
        a_c_i = df_c_i['Cubist-Impressionist-A'][1:97]
        b_c_i = df_c_i['Cubist-Impressionist-B'][1:97]
        consine_c_i = df_c_i['Cubist-Impressionist-Cosine'][1:97]
        accuracy_c_i = df_c_i['Percent-Correct'][1:97]

        a = []
        a.extend(a_c_i)

        b = []
        b.extend(b_c_i)

        consine = []
        consine.extend(consine_c_i)

        accuracy = []
        accuracy.extend(accuracy_c_i)
        self.alpha_pairs['diff'] = [a, b, consine, accuracy]
        #print (self.alpha_pairs['diff'])


        # beta same
        a = df_c_c['Cubist-Cubist-A'][49:]
        a_c_c = df_c_c['Cubist-Cubist-A'][49:]
        b_c_c = df_c_c['Cubist-Cubist-B'][49:]
        consine_c_c = df_c_c['Cubist-Cubist-Cosine'][49:]
        accuracy_c_c = df_c_c['Percent-Correct'][49:]

        a_i_i = df_i_i['Impressionist-Impressionist-A'][49:]
        b_i_i = df_i_i['Impressionist-Impressionist-B'][49:]
        consine_i_i = df_i_i['Impressionist-Impressionist-Cosine'][49:]
        accuracy_i_i = df_i_i['Percent-Correct'][49:]

        a = []
        a.extend(a_c_c)
        a.extend(a_i_i)

        b = []
        b.extend(b_c_c)
        b.extend(b_i_i)

        consine = []
        consine.extend(consine_c_c)
        consine.extend(consine_i_i)

        accuracy = []
        accuracy.extend(accuracy_c_c)
        accuracy.extend(accuracy_i_i)
        self.beta_pairs['same'] = [a, b, consine, accuracy]
        #print (self.beta_pairs['same'])

        # beta diff
        a_c_i = df_c_i['Cubist-Impressionist-A'][97:]
        b_c_i = df_c_i['Cubist-Impressionist-B'][97:]
        consine_c_i = df_c_i['Cubist-Impressionist-Cosine'][97:]
        accuracy_c_i = df_c_i['Percent-Correct'][97:]

        a = []
        a.extend(a_c_i)

        b = []
        b.extend(b_c_i)

        consine = []
        consine.extend(consine_c_i)

        accuracy = []
        accuracy.extend(accuracy_c_i)
        self.beta_pairs['diff'] = [a, b, consine, accuracy]
        #print (self.beta_pairs['diff'])
        
