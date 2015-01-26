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
	This fuction uses the element-wise array operations of numpy
	to get the mantissas of each number's log.

	arr: np.array of integers or floats ---> np.array of floats
	'''

	return np.log10(arr) - np.log10(arr).astype(int)


def firstDigit(output_DF=True):
	'''
	Returns the expected probabilities of the first digits
	according to Benford's distribution.
	
	- output_DF: Defaluts to True, outputing a pandas DataFrame
				with the probabilities, and the respective
				digits as the index, or a numpy array if False, when
				there will not be an index, and one must remember
				that the indexing for the 1 probability will be 
				[0], [1] for the two, and so on up untill [8] for
				the probability of the first digit 9.
	'''
	First_Dig = np.arange(1,10)
	Expected = np.log10(1 + (1. / First_Dig))
	if output_DF == False:
		return Expected
	else:
		return pd.DataFrame({'Expected':Expected,\
			'First_Dig':First_Dig}).set_index('First_Dig')

def secondDigit(output_DF=True):
	'''
	Returns the expected probabilities of the second digits
	according to Benford's distribution.
	
	output_DF: Defaluts to Ture, Outputing a pandas DataFrame
			with the digit as index and the respective probability
			in the 'prob' column, or a numpy array if False. In
			this case, coincidently the indexing within the array
			will match the second digits -> [0] - prob. of 0,
			[1] - prob of 1, and sort	on up untill [9] - prob of 9.
	'''
	a = np.arange(10,100)
	Expected = np.log10(1 + (1. / a))
	Sec_Dig = np.array(range(10)*9)
	if output_DF == False:
		c = np.zeros(10)
		for n in Sec_Dig:
			c[n] = Expected[Sec_Dig == n].sum()
		return c
	else:
		d = pd.DataFrame({'Expected': Expected, 'Sec_Dig': Sec_Dig},\
		index = a)
		return d.groupby('Sec_Dig').agg(sum)


def firstTwoDigits(output_DF=True):
	'''
	Returns the expected probabilities of the first two digits
	according to Benford's distribution.
	
	- output_DF: Defaluts to Ture, Outputing a pandas DataFrame
				object with the probabilities and the respective
				digits as the index, or a numpy array if False.
	'''
	First_2_Dig = np.arange(10,100)
	Expected = np.log10(1 + (1. / First_2_Dig))
	if output_DF == False:
		return Expected
	else:
		return return pd.DataFrame({'First_2_Dig':First_2_Dig,\
			'Expected':Expected}).set_index('First_2_Dig')


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

