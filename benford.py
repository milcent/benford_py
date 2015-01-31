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



def __getMantissas__(arr):
	'''
	The mantissa is the non-integer part of the log of a number.
	This fuction uses the element-wise array operations of numpy
	to get the mantissas of each number's log.

	arr: np.array of integers or floats ---> np.array of floats
	'''

	return np.log10(arr) - np.log10(arr).astype(int)


def __first__(output_DF=True):
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

def __second__(output_DF=True):
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


def __firstTwo__(output_DF=True):
	'''
	Returns the expected probabilities of the first two digits
	according to Benford's distribution.
	
	- output_DF: Defaluts to Ture, Outputing a pandas DataFrame
				object with the probabilities and the respective
				digits as the index, or a numpy array if False, when
				there will not be an index column, and one must
				remember that the indexing for the 10 probability
				will be [0], [1] for 11, and so on up untill [89] for
				the probability of the first digit 99.
	'''
	First_2_Dig = np.arange(10,100)
	Expected = np.log10(1 + (1. / First_2_Dig))
	if output_DF == False:
		return Expected
	else:
		return pd.DataFrame({'First_2_Dig':First_2_Dig,\
			'Expected':Expected}).set_index('First_2_Dig')


def firstTwoDigits(arr, dropLowerTen=True, MAD=True, Z_test=True,\
	MSE=False, lowUpBounds=True, plot=True):
	'''
	Performs the First Two Digits test with the series of numbers provided.

	'''
	# Ensure we are dealing with a pandas Series object and no missing values
	arr = pd.Series(arr).dropna()
	# Handle numbers < 10
	if dropLowerTen == False:
		# Multiply by constant to make all number with at least two
		# digits at the left of the floating point.
		# Take the second [1] element, should the first be 0, invert it
		# and use the number of digits at the left to generate the power
		# to elevate 10
		p = len(str((1/arr[1]).astype(int))) + 1	
		arr *= 10**p
		print "The whole sequence was multiplied by " + str(10**p)\
		+ " to ensure that there is no number lower than ten left."
	else:
		n = len(arr[arr<10])			# number of values < 10
		p = float(n)/len(arr) * 100		# and their proportion
		arr = arr[arr>=10]				# Discard all < 10
		print "Discarded " + str(n) + " values lower than 10, corresponding to "\
		+ str(p) + " percent of the sample."
	# convert into string, take the first two digits, and then convert
	# back to integer 		
	arr = arr.apply(str).apply(lambda x: x[:2]).apply(int)
	# get the number of occurrences of the first two digits
	v = arr.value_counts()
	# get their relative frequencies
	p = arr.value_counts(normalize =True)
	# crate dataframe from them
	df = pd.DataFrame({'Counts': v, 'Found': p}).sort_index()
	# reindex from 10 to 99 if one of the first two digits are
	# missing so the Expected frequencies column can later be
	# joined, and fill NANs with 0
	if len(df.index<90):
		df = df.reindex(np.arange(10,100)).fillna(0)
	# join the dataframe with the one of expected Benford's frequencies
	df = __firstTwo__().join(df)
	N = len(arr)
	# create column with absolute differences
	df['AbsDif'] = np.absolute(df.Found - df.Expected)
	# calculate the Z-test colum
	if Z_test == True:
		df['Z_test'] = (df.AbsDif - (1/2*N))/(np.sqrt(df.Expected*\
		(1-df.Expected/N)))
	# Mean absolute difference
	if MAD == True:
		mad = df.AbsDif.mean()
		print "Mean Absolute Deviation = " + str(mad)
	#Mean Square Error
	if MSE == False:
		mse = (df.AbsDif**2).mean()
		print "Mean Square Error = " + str(mse)



