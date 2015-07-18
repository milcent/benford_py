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

class First(pd.DataFrame):
 	"""	Returns the expected probabilities of the first digits
	according to Benford's distribution."""

 	def __init__(self, plot = True):
 		First_Dig = np.arange(1,10)
 		Exp = np.log10(1 + (1. / First_Dig))

 		pd.DataFrame.__init__(self, {'Expected':\
 			Exp}, index = First_Dig)
 		self.index.names = ['First_Dig']

		if plot == True:
			self.plot(kind='bar', color = 'g', grid=False)

class Second(pd.DataFrame):
	'''
	Returns the expected probabilities of the second digits
	according to Benford's distribution.
	'''
	def __init__(self, plot = True):
		a = np.arange(10,100)
		Expe = np.log10(1 + (1. / a))
		Sec_Dig = np.array(range(10)*9)

		pd.DataFrame.__init__(self,{'Expected': Expe, 'Sec_Dig': Sec_Dig})#index = a)
		self = self.groupby('Sec_Dig').sum()
		if plot == True:
			self.plot(kind='bar', color = 'g', grid=False, ylim=(0,.14))

class FirstTwo(pd.DataFrame):
	'''
	Returns the expected probabilities of the first two digits
	according to Benford's distribution.
	'''
	def __init__(self, plot=True):
		First_2_Dig = np.arange(10,100)
		Expect = np.log10(1 + (1. / First_2_Dig))

		pd.DataFrame.__init__(self,{'Expected':Expect, 'First_2_Dig':\
			First_2_Dig})
		self.set_index('First_2_Dig', inplace=True)
		if plot == True:
			self.plot(kind='bar', figsize = (15,8), color='g', grid=False)

class LastTwo(pd.DataFrame):
	'''   
	Returns the expected probabilities of the last two digits
	according to Benford's distribution.
	'''
	def __init__(self, plot=True):
		exp = np.array([1/99.]*100)
		pd.DataFrame.__init__(self,{'Expected': exp,'Last_2_Dig':_lt_()})
		self.set_index('Last_2_Dig', inplace=True)
		if plot == True:
			self.plot(kind='bar',figsize = (15,8), color = 'g',\
				grid=False,  ylim=(0,.02))

class Analysis(pd.DataFrame):

	maps = {}

	def __init__(self, data):
		pd.DataFrame.__init__(self, {'Seq': data})
		self.dropna(inplace=True)
		print "Initialized sequence with " + str(len(self)) + " registries."

	def mantissas(self, plot=True, figsize=(15,8)):
		# if self.Seq.dtype != 'Float64':
		# 	self.apply(float)
		self['Mant'] = _getMantissas_(self.Seq)
		self.sort('Mant', inplace+True)
		N = len(self)

		
		f = lambda g:g/N
		x = np.arange(N)
		fig = plt.figure(figsize=figsize)
		ax = fig.add_subplot(111)
		ax.plot(x,self.Mant,'k--')
		ax.plot(x,f(x), 'b-')

	def prepare(self, dec=2):
		'''
		Prepares the DataFrame to be manipulated by the tests, with columns
		of the First, Second, First Two and Last Two digits of each number
		'''
		# Extracts the digits in their respective positions,
		self.Seq = self.Seq * (10**dec)
		self.Seq = self.Seq.apply(_tint_)
		self = self[self.Seq!=0]
		ST = self.Seq.apply(str)
		self['FD'] = ST.apply(lambda x: x[:1])   # get the first digit
		self['SD'] = ST.apply(lambda x: x[1:2])  # get the second digit
		self['FTD'] = ST.apply(lambda x: x[:2])  # get the first two digits
		
		self['LTD'] = ST.apply(lambda x: x[-2:]) # get the last two digits
		# Leave the last two digits as strings , so as to be able to\
		# display '00', '01', ... up to '09', till '99'
		# converting the others to integers
		self[['FD','SD','FTD']].apply(int)

		self = self[self.FTD>=10]


	def firstTwoDigits(self, inform=True, MAD=True, Z_test=True, top_Z=20, MSE=False, plot=True,\
		mantissa = False):
		'''
		Performs the Benford First Two Digits test with the series of
		numbers provided.

		MAD -> calculates the Mean of the Absolute Differences from the respective
		expected distributions; defaults to True.

		Z_test -> calculates the Z test of the sample; defaluts to True.

		top_Z -> chooses the highest number of Z scores to be displayed

		MSE -> calculate the Mean Square Error of the sample; defaluts to False.

		plot -> draws the plot of test for visual comparison, with the found
		distributions in bars and the expected ones in a line.

		'''
		N = len(self)
		x = np.arange(10,100)
		if inform:
			print "\n---Test performed on " + str(N) + " registries.---\n"
		# get the number of occurrences of the first two digits
		v = self.FTD.value_counts()
		# get their relative frequencies
		p = self.FTD.value_counts(normalize =True)
		# crate dataframe from them
		df = pd.DataFrame({'Counts': v, 'Found': p}).sort_index()
		# reindex from 10 to 99 in the case one or more of the first
		# two digits are missing, so the Expected frequencies column
		# can later be joined; and swap NANs with zeros.
		if len(df.index) < 90:
			df = df.reindex(x).fillna(0)
		# join the dataframe with the one of expected Benford's frequencies
		df = FirstTwo(plot=False).join(df)
		# create column with absolute differences
		df['AbsDif'] = np.absolute(df.Found - df.Expected)
		# calculate the Z-test column an display the dataframe by descending
		# Z test
		if Z_test == True:
			df['Z_test'] = _Z_test(df,N)	
			print '\nThe top ' + str(top_Z) + ' Z scores are:\n'
			print df[['Expected','Found','Z_test']].sort('Z_test',\
			 ascending=False).head(top_Z)
		# Mean absolute difference
		if MAD == True:
			mad = _mad_(df)
			print "\nThe Mean Absolute Deviation is " + str(mad) + '\n'\
			+ 'For the First Two Digits:\n\
			- 0.0000 to 0.0012: Close Conformity\n\
			- 0.0012 to 0.0018: Acceptable Conformity\n\
			- 0.0018 to 0.0022: Marginally Acceptable Conformity\n\
			- Above 0.0022: Nonconformity'
		# Mean Square Error
		if MSE == True:
			mse = _mse_(df)
			print "\nMean Square Error = " + str(mse)
		# Plotting the expected frequncies (line) against the found ones(bars)
		if plot == True:
			_plot_benf_(df, x=x, y_Exp= df.Expected,y_Found=df.Found, N=N)

		if mantissa == True:
			df['Mantissas'] = np.log10(g) - np.log10(g).astype(int)

		return df

	def firstDigit(self, inform=True, MAD=True, Z_test=True, MSE=False, plot=True):
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
		x = np.arange(1,10)
		if inform:
			print "\n---Test performed on " + str(N) + " registries.---\n"
		# get the number of occurrences of each first digit
		v = self.FD.value_counts()
		# get their relative frequencies
		p = self.FD.value_counts(normalize =True)
		# crate dataframe from them
		df = pd.DataFrame({'Counts': v, 'Found': p}).sort_index()
		# reindex from 10 to 99 in the case one or more of the first
		# two digits are missing, so the Expected frequencies column
		# can later be joined; and swap NANs with zeros.

		# join the dataframe with the one of expected Benford's frequencies
		df = First(plot=False).join(df)
		# create column with absolute differences
		df['AbsDif'] = np.absolute(df.Found - df.Expected)
		# calculate the Z-test column an display the dataframe by descending
		# Z test
		if Z_test == True:
			df['Z_test'] = _Z_test(df,N)
			print '\nThe highest Z scores are:\n'
			print df[['Expected','Found','Z_test']].sort('Z_test',\
			 ascending=False)
		# Mean absolute difference
		if MAD == True:
			mad = _mad_(df)
			print "\nThe Mean Absolute Deviation is " + str(mad) + '\n'\
			+ 'For the First DigitS:\n\
			- 0.0000 to 0.0006: Close Conformity\n\
			- 0.0006 to 0.0012: Acceptable Conformity\n\
			- 0.0012 to 0.0015: Marginally Acceptable Conformity\n\
			- Above 0.0015: Nonconformity'
		# Mean Square Error
		if MSE == True:
			mse = _mse_(df)
			print "\nMean Square Error = " + str(mse)
		# Plotting the expected frequncies (line) against the found ones(bars)
		if plot == True:
			_plot_benf_(df, x=x, y_Exp= df.Expected,y_Found=df.Found, N=N)

		return df

	def secondDigit(self, inform=True, MAD=True, Z_test=True, MSE=False, plot=True):
		'''
		Performs the Benford Second Digit test with the series of
		numbers provided.

		MAD -> calculates the Mean of the Absolute Differences from the respective
		expected distributions; defaults to True.

		Z_test -> calculates the Z test of the sample; defaluts to True.

		MSE -> calculate the Mean Square Error of the sample; defaluts to False.

		plot -> draws the plot of test for visual comparison, with the found
		distributions in bars and the expected ones in a line.

		'''

		N = len(self)
		x = np.arange(0,10)
		if inform:
			print "\n---Test performed on " + str(N) + " registries.---\n"
		# get the number of occurrences of each second digit
		v = self.SD.value_counts()
		# get their relative frequencies
		p = self.SD.value_counts(normalize =True)
		# crate dataframe from them
		d = pd.DataFrame({'Counts': v, 'Found': p}).sort_index()
		# reindex from 10 to 99 in the case one or more of the first
		# two digits are missing, so the Expected frequencies column
		# can later be joined; and swap NANs with zeros.

		# join the dataframe with the one of expected Benford's frequencies
		df = Second(plot=False).groupby('Sec_Dig').sum().join(d)
		# create column with absolute differences
		df['AbsDif'] = np.absolute(df.Found - df.Expected)
		# calculate the Z-test column an display the dataframe by descending
		# Z test
		if Z_test == True:
			df['Z_test'] = _Z_test(df,N)
			print '\nThe highest Z scores are:\n'
			print df[['Expected','Found','Z_test']].sort('Z_test',\
			 ascending=False)
		# Mean absolute difference
		if MAD == True:
			mad = _mad_(df)
			print "\nThe Mean Absolute Deviation is " + str(mad) + '\n'\
			+ 'For the Second DigitS:\n\
			- 0.0000 to 0.0008: Close Conformity\n\
			- 0.0008 to 0.0010: Acceptable Conformity\n\
			- 0.0010 to 0.0012: Marginally Acceptable Conformity\n\
			- Above 0.0012: Nonconformity'
		# Mean Square Error
		if MSE == True:
			mse = _mse_(df)
			print "\nMean Square Error = " + str(mse)
		# Plotting the expected frequncies (line) against the found ones(bars)

		if plot == True:
			_plot_benf_(df, x=x, y_Exp= df.Expected,y_Found=df.Found, N=N)

		return df

	def lastTwoDigits(self, inform=True, MAD=False, Z_test=True, top_Z=20, MSE=False, plot=True):
		'''
		Performs the Benford Last Two Digits test with the series of
		numbers provided.

		MAD -> calculates the Mean of the Absolute Differences from the respective
		expected distributions; defaults to True.

		Z_test -> calculates the Z test of the sample; defaluts to True.

		top_Z -> chooses the highest number of Z scores to be displayed

		MSE -> calculate the Mean Square Error of the sample; defaluts to False.

		plot -> draws the plot of test for visual comparison, with the found
		distributions in bars and the expected ones in a line.

		'''

		N = len(self)
		x = np.arange(0,100)
		if inform:
			print "\n---Test performed on " + str(N) + " registries.---\n"
		# get the number of occurrences of the last two digits
		v = self.LTD.value_counts()
		# get their relative frequencies
		p = self.LTD.value_counts(normalize =True)
		# crate dataframe from them
		df = pd.DataFrame({'Counts': v, 'Found': p}).sort_index()
		# join the dataframe with the one of expected Benford's frequencies
		df = LastTwo(plot=False).join(df)
		# create column with absolute differences
		df['AbsDif'] = np.absolute(df.Found - df.Expected)
		# calculate the Z-test column an display the dataframe by descending
		# Z test
		if Z_test == True:
			df['Z_test'] = _Z_test(df,N)
			print '\nThe top ' + str(top_Z) +' Z scores are:\n'
			print df[['Expected','Found','Z_test']].sort('Z_test',\
			 ascending=False).head(top_Z)
		# Mean absolute difference
		if MAD == True:
			mad = _mad_(df)
			print "\nThe Mean Absolute Deviation is " + str(mad) + '\n'\
		# 	+ 'For the Second DigitS:\n\
		# 	- 0.0000 to 0.0008: Close Conformity\n\
		# 	- 0.0008 to 0.0010: Acceptable Conformity\n\
		# 	- 0.0010 to 0.0012: Marginally Acceptable Conformity\n\
		# 	- Above 0.0012: Nonconformity'
		# Mean Square Error
		if MSE == True:
			mse = _mse_(df)
			print "\nMean Square Error = " + str(mse)
		# Plotting the expected frequncies (line) against the found ones(bars)

		if plot == True:
			_plot_benf_(df, x=x, y_Exp= df.Expected,y_Found=df.Found, N=N)

		return df
	
	def duplicates(self, inform=True, top_Rep=20):
		# self.Seq = self.Seq.apply(int) / 100.
		N = len(self)
		self.Seq = self.Seq.apply(_to_float_)
		# get the frequencies
		v = self.Seq.value_counts()
		# get their relative frequencies
		p = self.Seq.value_counts(normalize =True) * 100
		# crate dataframe from them
		df = pd.DataFrame({'Counts': v, 'Percent': p}).sort('Counts',\
			ascending=False)
		if inform:
			print "\n---Test performed on " + str(N) + " registries.---\n"
			print '\nThe ' + str(top_Rep) + ' most frequent numbers are:\n'
			print df.head(top_Rep)
		return df

def _Z_test(frame,N):
	return (frame.AbsDif - (1/2*N))/(np.sqrt(frame.Expected*\
		(1-frame.Expected)/N))

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


def _first_(plot=False):
	'''
	Returns the expected probabilities of the first digits
	according to Benford's distribution.
	'''
	First_Dig = np.arange(1,10)
	Expected = np.log10(1 + (1. / First_Dig))
	first = pd.DataFrame({'Expected':Expected,\
			'First_Dig':First_Dig}).set_index('First_Dig')
	if plot == True:
		first.plot(kind='bar', grid=False)
	return first

def _second_(plot=False):
	'''
	Returns the expected probabilities of the second digits
	according to Benford's distribution.
	'''
	a = np.arange(10,100)
	Expected = np.log10(1 + (1. / a))
	Sec_Dig = np.array(range(10)*9)
	d = pd.DataFrame({'Expected': Expected, 'Sec_Dig': Sec_Dig},\
			index = a)
	sec = d.groupby('Sec_Dig').agg(sum)
	if plot == True:
		sec.plot(kind='bar', grid=False)
	return sec

def _firstTwo_(plot=False):
	'''
	Returns the expected probabilities of the first two digits
	according to Benford's distribution.
	'''
	First_2_Dig = np.arange(10,100)
	Expected = np.log10(1 + (1. / First_2_Dig))
	ft = pd.DataFrame({'First_2_Dig':First_2_Dig,\
			'Expected':Expected}).set_index('First_2_Dig')
	if plot == True:
		ft.plot(kind='bar', figsize = (15,8),grid=False)
	return ft

def _lastTwo_(plot=False):
	exp = np.array([1/99.]*100)
	lt = pd.DataFrame({'Last_2_Dig': _lt_(),\
			'Expected': exp}).set_index('Last_2_Dig')
	if plot == True:
		lt.plot(kind='bar',figsize = (15,8), grid=False,  ylim=(0,.02))
	return lt

def _lt_():
	l = []
	d = '0123456789'
	for i in d:
		for j in d:
			t = i+j
			l.append(t)
	return np.array(l)

def _to_float_(st):
	try:
		return float(st) /100
	except:
		return np.nan

def _sanitize_(arr):
	'''
	Prepares the series to enter the test functions, in case pandas
	has not inferred the type to be float, especially when parsing
	from latin datases which use '.' for thousands and ',' for the
	floating point.
	'''
	return pd.Series(arr).dropna().apply(str).apply(_only_numerics_).apply(_l_0_strip_)

	#if not isinstance(arr[0:1],float):
	#	arr = arr.apply(str).apply(lambda x: x.replace('.','')).apply(lambda x:\
	#	 x.replace(',','.')).apply(float)
	#return arr.abs()

def _only_numerics_(seq):
    return filter(type(seq).isdigit, seq)

def _str_to_float_(s):
	#s = str(s)
	if '.' in s or ',' in s:
		s = filter(type(s).isdigit, s)
		s = s[:-2]+'.'+s[-2:]
		return float(s)
	else:
		if filter(type(s).isdigit, s) == '':
			return np.nan
		else:
			return int(s)


def _l_0_strip_(st):
	return st.lstrip('0')

def _tint_(s):
	try:
		return int(s)
	except:
		return 0

def _len2_(st):
	return len(st) == 2

def _plot_benf_(df, x, y_Exp, y_Found, N,lowUpBounds = True, figsize=(15,8)):		
	fig = plt.figure(figsize=figsize)
	ax = fig.add_subplot(111)
	plt.title('Expected vs. Found Distributions')
	plt.xlabel('Digits')
	plt.ylabel('Distribution')
	ax.bar(x, y_Found, label='Found')
	ax.plot(x, y_Exp, color='g',linewidth=2.5,\
	 label='Expected')
	ax.legend()
	# Plotting the Upper and Lower bounds considering p=0.05
	if lowUpBounds == True:
		sig_5 = 1.96 * np.sqrt(y_Exp*(1-y_Exp)/N)
		upper = y_Exp + sig_5 + (1/(2*N))
		lower = y_Exp - sig_5 - (1/(2*N))
		ax.plot(x, upper, color= 'r')
		ax.plot(x, lower, color= 'r')
		ax.fill_between(x, upper,lower, color='r', alpha=.3)
	plt.show()


def _collapse_(num):
	'''
	Transforms any positive number to the form XX.yy, with two digits
	to the left of the floating point
	'''
	l=10**int(np.log10(num))
	if num>=1.:
		return 10.*num/l
	else:
		return 100.*num/l

def _collapse_array_(arr):
	'''

	'''
	# arr = abs(arr)
	ilt = 10**(np.log10(arr).astype(int))
	arr[arr<1.]*=10
	return arr*10/ilt

def _sanitize_float_(s):
	s = str(s)
	if '.' in s or ',' in s:
		s = filter(type(s).isdigit, s)
		s = s[:-2]+'.'+s[-2:]
		return float(s)
	else:
		if filter(type(s).isdigit, s) == '':
			return np.nan
		else:
			return int(s)

