'''
Author: Marcel Milcent

This is a module for application of Benford's Law to a sequence of 
numbers.

Dependent on pandas and numpy, using matplotlib for visualization

All logarithms ar in base 10: "np.log10"
'''

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# create expected Benford distributions for the first digit, the 
# second digit, and the first two digits

# create functions to run the tests

def _get_mantissas_(arr):
	'''
	The mantissa is the non-integer part of the log of a number.
	This fuction uses the element-wise array operations on numpy
	to get the mantissas of each number's log.

	arr: numpy array of integers or floats
	'''

	return np.log10(arr) - np.log10(arr).astype(int)