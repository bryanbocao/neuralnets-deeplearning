'''
From https://github.com/tylersco/deep-neural-networks-art-images
Author: https://github.com/tylersco
'''
import numpy as np

def cos_sim(a, b, threshold=0.00000001):
	dot_product = np.dot(a, b)
	norm_a = np.linalg.norm(a)
	norm_b = np.linalg.norm(b)

	if np.abs(norm_a) < threshold or np.abs(norm_b) < threshold:
		return 0

	return dot_product / (norm_a * norm_b)
