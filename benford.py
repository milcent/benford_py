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

def _getMantissas_(arr):
	'''
	The mantissa is the non-integer part of the log of a number.
	This fuction uses the element-wise array operations on numpy
	to get the mantissas of each number's log.

	arr: array of integers or floats >>> array of floats
	'''

	return np.log10(arr) - np.log10(arr).astype(int)

# create arrays with expected Benford's distributions

def firstDigit():
	a = np.arange(1,10)
	return np.log10(1 + (1. / a))


def secondDigit(in_dataFrame=True):
	a = np.arange(10,100)
	probs = np.log10(1 + (1. / a))
	s = np.array(range(10)*9)
	if in_dataFrame == False:
		c = np.zeros(10)
		for n in b:
			c[n] = probs[s == n].sum()
		return c
	else:
		d = pd.DataFrame({'probs': probs, 'secdig': s}, index = a)
		return d.groupby('secdig').agg(sum)


def firstTwoDigits():
	a = np.arange(10,100)
	return np.log10(1 + (1. / a))


def lowUpBounds():

def ________(arr, digits=2, dropLowerTen=True):
	arr.sort()
	if dropLowerTen == False:
		# Multiply by constant to make all number with at least two
		# digits left from the floating point.
		# Take the second [1] element, should the first be 0, invert it
		# and use the number of digits at the left to generate the power
		# to elevate 10
		p = len(str((1/arr[1]).astype(int)))	
		arr *= 10**(p+1)
	else:
		arr = arr[arr>=10]

