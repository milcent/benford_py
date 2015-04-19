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


class Benford(pd.DataFrame):

	results = {}

	def __init__(self, data):
		pd.DataFrame.__init__(self, {'Seq': _sanitize_(data)})
		print "Initialized sequence with " + str(len(self)) + " registries."

	def firstTwoDigits(self, dropLowerTen=True, MAD=True, Z_test=True,\
		MSE=False, plot=True):
		'''
		Performs the Benford First Two Digits test with the series of
		numbers provided.

		dropLowerTen ->  option to discard numbers lower than 10, so as to 
		keep the tested numbers with two first digits; defaults to True.

		MAD -> calculates the Mean of the Absolute Differences from the respective
		expected distributions; defaults to True.

		Z_test -> calculates the Z test of the sample; defaluts to True.

		MSE -> calculate the Mean Square Error of the sample; defaluts to False.

		plot -> draws the plot of test for visual comparison, with the found
		distributions in bars and the expected ones in a line.

		'''
		# Handle numbers < 10
		if dropLowerTen == False:
			# Multiply by constant to make all numbers with at least two
			# digits at the left of the floating point.
			# Take the second [1] element, should the first be 0, invert it
			# and use the number of digits at the left to generate the power
			# to elevate 10
			p = len(str((1/self.Seq[1:2]).astype(int))) + 1	
			self *= 10**p
			print "---The whole sequence was multiplied by " + str(10**p)\
			+ " to ensure that there is no number lower than ten left.---"
		else:
			n = len(self[self.Seq<10])			# number of values < 10
			p = float(n)/len(self) * 100		# and their proportion
			self = self[self.Seq>=10]				# Discard all < 10
			print "---Discarded " + str(n) + " values lower than 10, corresponding to "\
			+ str(np.round(p,2)) + " percent of the sample.---"
		N = len(self)
		print "\n---Performing test on " + str(N) + " registries.---\n"
		# convert into string, take the first two digits, and then convert
		# back to integer 		
		self.Seq = self.Seq.apply(str).apply(lambda x: x[:2]).apply(int)
		# get the number of occurrences of the first two digits
		v = self.Seq.value_counts()
		# get their relative frequencies
		p = self.Seq.value_counts(normalize =True)
		# crate dataframe from them
		df = pd.DataFrame({'Counts': v, 'Found': p}).sort_index()
		# reindex from 10 to 99 in the case one or more of the first
		# two digits are missing, so the Expected frequencies column
		# can later be joined; and swap NANs with zeros.
		if len(df.index) < 90:
			df = df.reindex(np.arange(10,100)).fillna(0)
		# join the dataframe with the one of expected Benford's frequencies
		df = _firstTwo_().join(df)
		# create column with absolute differences
		df['AbsDif'] = np.absolute(df.Found - df.Expected)
		# calculate the Z-test column an display the dataframe by descending
		# Z test
		if Z_test == True:
			df['Z_test'] = _Z_test(df,N)	
			print '\nThe 15 highest Z scores are:\n'
			print df[['Expected','Found','Z_test']].sort('Z_test',\
			 ascending=False).head(15)
		# Mean absolute difference
		if MAD == True:
			mad = _mad_(df)
			print "\nThe Mean Absolute Deviation is " + str(mad) + '\n'\
			+ 'For the First Two Digits:\n\
			- 0.0000 to 0.0012: Close Conformity\n\
			- 0.0012 to 0.0018: Acceptable Conformity\n\
			- 0.0018 to 0.0022: Marginally Acceptable Conformity\n\
			- > 0.0022: Nonconformity'
		# Mean Square Error
		if MSE == True:
			mse = _mse_(df)
			print "\nMean Square Error = " + str(mse)
		# Plotting the expected frequncies (line) against the found ones(bars)
		if plot == True:
			_plot_benford_(df, N)

		return df

	def firstDigit(self, MAD=True, Z_test=True, MSE=False, plot=True):
		'''
		Performs the Benford First Digit test with the series of
		numbers provided.

		MAD -> calculates the Mean of the Absolute Differences from the respective
		expected distributions; defaults to True.

		Z_test -> calculates the Z test of the sample; defaluts to True.

		MSE -> calculate the Mean Square Error of the sample; defaluts to False.

		plot -> draws the plot of test for visual comparison, with the found
		distributions in bars and the expected ones in a line.

		'''

		N = len(self)
		print "\n---Performing test on " + str(N) + " registries.---\n"
		# convert into string, take the first two digits, and then convert
		# back to integer 		
		self.Seq = self.Seq *100
		self.Seq.apply(str).apply(lambda x: x[:1]).apply(int)
		# get the number of occurrences of the first two digits
		v = self.Seq.value_counts()
		# get their relative frequencies
		p = self.Seq.value_counts(normalize =True)
		# crate dataframe from them
		df = pd.DataFrame({'Counts': v, 'Found': p}).sort_index()
		# reindex from 10 to 99 in the case one or more of the first
		# two digits are missing, so the Expected frequencies column
		# can later be joined; and swap NANs with zeros.
		if len(df.index) < 9:
			df = df.reindex(np.arange(1,10)).fillna(0)
		# join the dataframe with the one of expected Benford's frequencies
		df = _first_().join(df)
		# create column with absolute differences
		df['AbsDif'] = np.absolute(df.Found - df.Expected)
		# calculate the Z-test column an display the dataframe by descending
		# Z test
		if Z_test == True:
			df['Z_test'] = _Z_test(df,N)	
		# Mean absolute difference
		if MAD == True:
			mad = _mad_(df)
			print "\nThe Mean Absolute Deviation is " + str(mad) + '\n'\
			+ 'For the First DigitS:\n\
			- 0.0000 to 0.0006: Close Conformity\n\
			- 0.0006 to 0.0012: Acceptable Conformity\n\
			- 0.0012 to 0.0015: Marginally Acceptable Conformity\n\
			- More than 0.0015: Nonconformity'
		# Mean Square Error
		if MSE == True:
			mse = _mse_(df)
			print "\nMean Square Error = " + str(mse)
		# Plotting the expected frequncies (line) against the found ones(bars)
		if plot == True:
			_plot_benford_(df, N)

		return df

def _Z_test(frame,N):
	return (frame.AbsDif - (1/2*N))/(np.sqrt(frame.Expected*\
		(1-frame.Expected)/N))
	print '\nThe highest Z scores are:\n'
	print frame[['Expected','Found','Z_test']].sort('Z_test',\
		 ascending=False).head(10)

def _mad_(frame):
	return frame.AbsDif.mean()

def _mse_(frame):
	return (frame.AbsDif**2).mean()


def _getMantissas_(arr):
	'''
	The mantissa is the non-integer part of the log of a number.
	This fuction uses the element-wise array operations of numpy
	to get the mantissas of each number's log.

	arr: np.array of integers or floats ---> np.array of floats
	'''

	return np.log10(arr) - np.log10(arr).astype(int)


def _first_():
	'''
	Returns the expected probabilities of the first digits
	according to Benford's distribution.
	'''
	First_Dig = np.arange(1,10)
	Expected = np.log10(1 + (1. / First_Dig))
	return pd.DataFrame({'Expected':Expected,\
			'First_Dig':First_Dig}).set_index('First_Dig')

def _second_():
	'''
	Returns the expected probabilities of the second digits
	according to Benford's distribution.
	'''
	a = np.arange(10,100)
	Expected = np.log10(1 + (1. / a))
	Sec_Dig = np.array(range(10)*9)
	d = pd.DataFrame({'Expected': Expected, 'Sec_Dig': Sec_Dig},\
			index = a)
	return d.groupby('Sec_Dig').agg(sum)


def _firstTwo_():
	'''
	Returns the expected probabilities of the first two digits
	according to Benford's distribution.
	'''
	First_2_Dig = np.arange(10,100)
	Expected = np.log10(1 + (1. / First_2_Dig))
	return pd.DataFrame({'First_2_Dig':First_2_Dig,\
			'Expected':Expected}).set_index('First_2_Dig')


def _sanitize_(arr):
	'''
	Prepares the series to enter the test functions, in case pandas
	has not inferred the type to be float, especially when parsing
	from latin datases which use '.' for thousands and ',' for the
	floating point.
	'''
	arr = pd.Series(arr).dropna()

	if not isinstance(arr[0:1],float):
		arr = arr.apply(str).apply(lambda x: x.replace('.','')).apply(lambda x:\
		 x.replace(',','.')).apply(float)

	return arr



def _plot_benford_(df, N,lowUpBounds = True):		
	fig = plt.figure(figsize=(15,10))
	ax = fig.add_subplot(111)
	plt.title('Expected versus Found Distributions')
	plt.xlabel('First Two Digits')
	plt.ylabel('Distribution')
	ax.bar(df.index, df.Found, label='Found')
	ax.plot(df.index,df.Expected, color='g',linewidth=2.5,\
	 label='Expected')
	ax.legend()
	# Plotting the Upper and Lower bounds considering p=0.05
	if lowUpBounds == True:
		sig_5 = 1.96 * np.sqrt(df.Expected*(1-df.Expected)/N)
		upper = df.Expected + sig_5 + (1/(2*N))
		lower = df.Expected - sig_5 - (1/(2*N))
		ax.plot(df.index, upper, color= 'r')
		ax.plot(df.index, lower, color= 'r')
		ax.fill_between(df.index, upper,lower, color='r', alpha=.3)
	plt.show()



