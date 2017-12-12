'''
From https://github.com/tylersco/deep-neural-networks-art-images
Author: https://github.com/tylersco
'''
import os
import re
import numpy as np
from scipy import misc
import pandas as pd
from sklearn.model_selection import train_test_split

class ArtData:

    def __init__(self, path):
        self.path = path
        self.image_folder = '16x16-small'
        self.metadata_filename = 'Corbett-PaintingPairs-CosineAndPercentCorrect.csv'
        self.image_size = 16

        self.train_images = {}
        self.image_pairs = []
        self.alpha_pairs = {
            'same': [],
            'diff': []
        }
        self.beta_pairs = {
            'same': [],
            'diff': []
        }

    def load_images_and_pairs(self):
        self.train_images = self.load(os.path.join(self.path, self.image_folder))

        metadata = pd.read_csv(os.path.join(self.path, self.metadata_filename))

        # Extract cubist-cubist pairs

        df_cubist_cubist = metadata[['Cubist-Cubist A', 'Cubist-Cubist B', 'Cubist-Cubist Cosine', 'Percent Correct']]
        df_cubist_cubist = df_cubist_cubist.dropna()

        alpha, beta = train_test_split(df_cubist_cubist, test_size=0.5)

        for i, row in alpha.iterrows():
            self.alpha_pairs['same'].append((self.train_images[row['Cubist-Cubist A']][0], self.train_images[row['Cubist-Cubist B']][0],
                row['Cubist-Cubist Cosine'], float(row['Percent Correct'][:-1]), row['Cubist-Cubist A'], row['Cubist-Cubist B']
            ))

        for i, row in beta.iterrows():
            self.beta_pairs['same'].append((self.train_images[row['Cubist-Cubist A']][0], self.train_images[row['Cubist-Cubist B']][0],
                row['Cubist-Cubist Cosine'], float(row['Percent Correct'][:-1]), row['Cubist-Cubist A'], row['Cubist-Cubist B']
            ))

        for i, row in df_cubist_cubist.iterrows():
            self.image_pairs.append((self.train_images[row['Cubist-Cubist A']][0], self.train_images[row['Cubist-Cubist B']][0],
                row['Cubist-Cubist Cosine'], float(row['Percent Correct'][:-1]), row['Cubist-Cubist A'], row['Cubist-Cubist B']
            ))

        # Extract impressionist-impressionist pairs

        df_impr_impr = metadata[['Impressionist-Impressionist A', 'Impressionist-Impressionist B', 'Impressionist-Impressionist Cosine', 'Percent Correct.1']]
        df_impr_impr = df_impr_impr.dropna()

        alpha, beta = train_test_split(df_impr_impr, test_size=0.5)

        for i, row in alpha.iterrows():
            self.alpha_pairs['same'].append((self.train_images[row['Impressionist-Impressionist A']][0], self.train_images[row['Impressionist-Impressionist B']][0],
                row['Impressionist-Impressionist Cosine'], float(row['Percent Correct.1'][:-1]), row['Impressionist-Impressionist A'], row['Impressionist-Impressionist B']
            ))

        for i, row in beta.iterrows():
            self.beta_pairs['same'].append((self.train_images[row['Impressionist-Impressionist A']][0], self.train_images[row['Impressionist-Impressionist B']][0],
                row['Impressionist-Impressionist Cosine'], float(row['Percent Correct.1'][:-1]), row['Impressionist-Impressionist A'], row['Impressionist-Impressionist B']
            ))

        for i, row in df_impr_impr.iterrows():
            self.image_pairs.append((self.train_images[row['Impressionist-Impressionist A']][0], self.train_images[row['Impressionist-Impressionist B']][0],
                row['Impressionist-Impressionist Cosine'], float(row['Percent Correct.1'][:-1]), row['Impressionist-Impressionist A'], row['Impressionist-Impressionist B']
            ))

        # Extract cubist-impressionist pairs

        df_cubist_impr = metadata[['Cubist-Impressionist A', 'Cubist-Impressionist B', 'Cubist-Impressionist Cosine', 'Percent Correct.2']]
        df_cubist_impr = df_cubist_impr.dropna()

        alpha, beta = train_test_split(df_cubist_impr, test_size=0.5)

        for i, row in alpha.iterrows():
            self.alpha_pairs['diff'].append((self.train_images[row['Cubist-Impressionist A']][0], self.train_images[row['Cubist-Impressionist B']][0],
                row['Cubist-Impressionist Cosine'], float(row['Percent Correct.2'][:-1]), row['Cubist-Impressionist A'], row['Cubist-Impressionist B'], -row['Cubist-Impressionist Cosine']
            ))

        for i, row in beta.iterrows():
            self.beta_pairs['diff'].append((self.train_images[row['Cubist-Impressionist A']][0], self.train_images[row['Cubist-Impressionist B']][0],
                row['Cubist-Impressionist Cosine'], float(row['Percent Correct.2'][:-1]), row['Cubist-Impressionist A'], row['Cubist-Impressionist B'], -row['Cubist-Impressionist Cosine']
            ))

        for i, row in df_cubist_impr.iterrows():
            self.image_pairs.append((self.train_images[row['Cubist-Impressionist A']][0], self.train_images[row['Cubist-Impressionist B']][0],
                row['Cubist-Impressionist Cosine'], float(row['Percent Correct.2'][:-1]), row['Cubist-Impressionist A'], row['Cubist-Impressionist B'], -row['Cubist-Impressionist Cosine']
            ))

        self.image_pairs = np.array(self.image_pairs)

    @classmethod
    def load(cls, path_img):
        images = {}
        for (dirpath, dirnames, filenames) in os.walk(path_img):
            for f in filenames:
                if not f.endswith('.jpg'):
                    continue

                with open(os.path.join(dirpath, f), 'rb') as file:
                    img = misc.imread(file)
                    img = np.array(img) / 255.0

                    # Extract label from filename
                    label = ''
                    l = re.search('cubist', f)
                    if l:
                        label = l.group(0)
                    else:
                        l = re.search('impressionist', f)
                        label = l.group(0)

                    images[f.replace('-small', '')] = (img, label)

        return images
