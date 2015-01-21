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

def _getMantissas_(arr):
	'''
	The mantissa is the non-integer part of the log of a number.
	This fuction uses the element-wise array operations on numpy
	to get the mantissas of each number's log.

	arr: array of integers or floats >>> array of floats
	'''

	return np.log10(arr) - np.log10(arr).astype(int)


def firstDigit(output_DF=True):
	'''
	Returns the expected probabilities of the first digits
	according to Benford's distribution.
	
	- output_DF: Defaluts to Ture, Outputing a pandas Series
				object with the probabilities and the respective
				digits as the index, or a numpy array if False.
	'''
	a = np.arange(1,10)
	b = np.log10(1 + (1. / a))
	if output_DF == False:
		return b
	else:
		return pd.Series(b, index = a)

def secondDigit(output_DF=True):
	'''
	Returns the expected probabilities of the second digits
	according to Benford's distribution.
	
	output_DF: Defaluts to Ture, Outputing a pandas DataFrame
	with the digit as index and the respective probability in
	the 'prob' column, or a numpy array if False.
	'''
	a = np.arange(10,100)
	prob = np.log10(1 + (1. / a))
	s = np.array(range(10)*9)
	if output_DF == False:
		c = np.zeros(10)
		for n in b:
			c[n] = prob[s == n].sum()
		return c
	else:
		d = pd.DataFrame({'prob': prob, 'secdig': s}, index = a)
		return d.groupby('secdig').agg(sum)


def firstTwoDigits(output_DF=True):
	'''
	Returns the expected probabilities of the first two digits
	according to Benford's distribution.
	
	- output_DF: Defaluts to Ture, Outputing a pandas Series
				object with the probabilities and the respective
				digits as the index, or a numpy array if False.
	'''
	a = np.arange(10,100)
	b = np.log10(1 + (1. / a))
	if output_DF == False:
		return b
	else:
		return pd.Series(b, index = a)


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

