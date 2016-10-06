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
 	'''
 	Returns the expected probabilities of the First, First Two, or
 	First Three digits according to Benford's distribution.

	-> digs: 1, 2 or 3 - tells which of the first digits to consider:
			1 for the First Digit, 2 for the First Two Digits and 3 for
			the First Three Digits.
	-> plot: option to plot a bar chart of the Expected proportions.
			Defaults to True.
	'''

 	def __init__(self, digs, plot = True):
 		if not digs in [1,2,3]:
 			raise ValueError("The value assigned to the parameter -digs-\
 was {0}. Value must be 1, 2 or 3.".format(digs))
 		dig_name = 'First_{0}_Dig'.format(digs)
 		Dig = np.arange(10**(digs-1),10**digs)
 		Exp = np.log10(1 + (1. / Dig))

 		pd.DataFrame.__init__(self, {'Expected': Exp}, index = Dig)
 		self.index.names = [dig_name]

		if plot == True:
			self.plot(kind='bar', color = 'g', grid=False,\
			 figsize=(5*(digs+1),4*(digs+.6)))

class Second(pd.DataFrame):
	'''
	Returns the expected probabilities of the Second Digits
	according to Benford's distribution.

	-> plot: option to plot a bar chart of the Expected proportions.
			Defaults to True.
	'''
	def __init__(self, plot = True):
		a = np.arange(10,100)
		Expe = np.log10(1 + (1. / a))
		Sec_Dig = np.array(range(10)*9)

		pd.DataFrame.__init__(self,{'Expected': Expe, 'Sec_Dig': Sec_Dig},\
		 index = a)
		self = self.groupby('Sec_Dig').sum()
		if plot == True:
			self.plot(kind='bar', color = 'g', grid=False,\
			figsize=(10,6.4), ylim=(0,.14))

class LastTwo(pd.DataFrame):
	'''   
	Returns the expected probabilities of the Last Two Digits
	according to Benford's distribution.

	-> plot: option to plot a bar chart of the Expected proportions.
			Defaults to True.
	'''
	def __init__(self, plot=True):
		exp = np.array([1/99.]*100)
		pd.DataFrame.__init__(self,{'Expected': exp,'Last_2_Dig':_lt_()})
		self.set_index('Last_2_Dig', inplace=True)
		if plot == True:
			self.plot(kind='bar',figsize = (15,8), color = 'g',\
				grid=False,  ylim=(0,.015))

class Analysis(pd.DataFrame):
	'''
	Initiates the Analysis of the series. pandas DataFrame subclass.
	Values must be integers or floats. If not, it will try to convert
	them. If it does not succeed, a TypeError will be raised.
	A pandas DataFrame will be constructed, with the columns: original
	numbers without floating points, first, second, first two, first three
	and	last two digits, so the following tests will run properly.

	-> data: sequence of numbers to be evaluated. Must be in absolute values,
			since negative values with minus signs will distort the tests.
			XXXXXX---PONDERAR DEIXAR O COMANDO DE CONVERTER PARA AbS------XXXXXX
	-> dec: number of decimal places to be accounted for. Especially important
			for the last two digits test. The numbers will be multiplied by
			10 to the power of the dec value. Defaluts to 2 (currency). If 
			the numbers are integers, assign 0.
	-> sec_order: choice for the Second Order Test, which cumputes the differences
			between the ordered entries before running the Tests.
	-> inform: tells the number of registries that are being subjected to
			the Analysis; defaults to True
	-> latin: used for str dtypes representing numbers in latin format, with
			'.' for thousands and ',' for decimals. Converts to a string with
			only '.' for decimals if float, and none if int, so it can be later
			converted to a number format. Defaults to False.
	'''
	maps = {} # dict for recording the indexes to be mapped back to the
			  # original series of numbers
	# dict of confidence levels for further use
	confs = {'None':None,'80':1.285,'85':1.435,'90':1.645,'95':1.96,'99':2.576,'99.9':3.29,\
	'99.99':3.89, '99.999':4.417, '99.9999':4.892, '99.99999':5.327} 
	digs_dict = {'1':'F1D','2':'F2D','3':'F3D'}

	def __init__(self, data, dec=2, sec_order=False, inform = True, latin=False):
		#if latin:
		#	thousands, decimals = '.', ','
		#else:
		#	thousands, decimals = ',', '.'
		pd.DataFrame.__init__(self, {'Seq': data}) #thousands = thousands, decimals = decimals
		self.dropna(inplace=True)
		if inform:
			print "Initialized sequence with {0} registries.".format(len(self)) 
		if self.Seq.dtypes != 'float' and self.Seq.dtypes != 'int':
			print 'Sequence dtype is not int nor float.\nTrying to convert...\n'
			if latin:
				if dec != 0:
					self.Seq = self.Seq.apply(_sanitize_latin_float_, dec=dec)
				else:
					self.Seq = self.Seq.apply(_sanitize_latin_int_)
			#Try to convert to numbers
			self.Seq = self.Seq.convert_objects(convert_numeric=True)
			self.dropna(inplace=True)
			if self.Seq.dtypes == 'float' or self.Seq.dtypes == 'int':
				print 'Conversion successful!'
			else:
				raise TypeError("The sequence dtype was not int nor float\
				 and could not be converted.\nConvert it to whether int of float,\
				  or set latin to True, and try again.")
		if sec_order:
			self.sort_values('Seq', inplace=True)
			self.drop_duplicates(inplace=True)
			self.Seq = self.Seq - self.Seq.shift(1)
			self.dropna(inplace=True)
			if inform:
				print 'Second Order Test. Initial series reduced to {0}\
 entries.'.format(len(self))
		# Extracts the digits in their respective positions,
		self['ZN'] = self.Seq * (10**dec)  # dec - to manage decimals
		if dec != 0:
			self.ZN = self.ZN.apply(_tint_)
		#self = self[self.ZN!=0]
		self['S'] = self.ZN.astype(str)
		self['F1D'] = self.S.str[:1]   # get the first digit
		self['SD'] = self.S.str[1:2]  # get the second digit
		self['F2D'] = self.S.str[:2]  # get the first two digits
		self['F3D'] = self.S.str[:3]  # get the first three digits
		self['L2D'] = self.S.str[-2:] # get the last two digits
		# Leave the last two digits as strings , so as to be able to\
		# display '00', '01', ... up to '09', till '99'
		# convert the others to integers
		self.F1D = self.F1D.apply(_tint_)
		self.SD = self.SD.apply(_tint_)
		self.F2D = self.F2D.apply(_tint_)
		self.F3D = self.F3D.apply(_tint_)
		del self['S']
		#self = self[self.F2D>=10]

	def mantissas(self, plot=True, figsize=(15,8)):
		'''
		Calculates the logs base 10 of the numbers in the sequence and Extracts
		the mantissae, which are the decimal parts of the logarithms. It them
		calculates the mean and variance of the mantissae, and compares them with
		the mean and variance of a Benford's sequence.
		plot -> plots the ordered mantissae and a line with the expected
				inclination. Defaults to True.
		figsize -> tuple that give sthe size of the figure when plotting
		'''
		self['Mant'] = _getMantissas_(self.Seq)
		p = self[['Seq','Mant']]
		p = p[p.Seq>0].sort_values('Mant')
		print "The Mantissas MEAN is {0}. Ref: 0.5.".format(p.Mant.mean())
		print "The Mantissas VARIANCE is {0}. Ref: 0.83333.".format(p.Mant.var())
		N = len(p)
		#return p
		if plot:
			p['x'] = np.arange(1,N+1)
			n = np.ones(N)/N
			fig = plt.figure(figsize=figsize)
			ax = fig.add_subplot(111)
			ax.plot(p.x,p.Mant,'r-', p.x,n.cumsum(),'b--',\
			 linewidth=2)
			plt.ylim((0,1.))
			plt.xlim((1,N+1))
			plt.show()
		return p

	def second_digit(self, inform=True, MAD=True, conf_level=95,\
		MSE=False, show_high_Z='pos', plot=True, ret = False):
		'''
		Performs the Benford Second Digit test with the series of
		numbers provided.

		inform -> tells the number of registries that are being subjected to
			the Analysis; defaults to True

		MAD -> calculates the Mean of the Absolute Differences between the found
			and the expected distributions; defaults to True.

		conf_level -> confidence level to draw lower and upper limits when
			plotting and to limit the mapping of the proportions to only the
			ones significantly diverging from the expected. Defaults to 95.
			If None, no boundaries will be drawn.

		show_high_Z -> chooses which Z scores to be used when displaying results,
			according to the confidence level chosen. Defaluts to 'pos', which will
			highlight only the values that are higher than the expexted frequencies;
			'all' will highlight both found extremes (positive and negative); and
			an integer, which will use the first n entries, positive and negative,
			regardless of whether the Z is higher than the conf_level Z or not

		MSE -> calculate the Mean Square Error of the sample; defaluts to False.

		plot -> draws the plot of test for visual comparison, with the found
			distributions in bars and the expected ones in a line.

		'''
		if str(conf_level) not in self.confs.keys():
			raise ValueError("Value of -conf_level- must be one of the\
 following: {0}".format(self.confs.keys()))
		conf = self.confs[str(conf_level)]
		N = len(self)
		x = np.arange(0,10)
		if inform:
			print "\n---Test performed on {0} registries.---\n".format(N)
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
		df['Dif'] = df.Found - df.Expected
		df['AbsDif'] = np.absolute(df.Dif)
		# calculate the Z-test column an display the dataframe by descending
		# Z test
		df['Z_test'] = _Z_test(df,N)
		
		#Populate dict with the most relevant entries
		self.maps['SD'] = np.array(_inform_and_map_(df, inform,\
		 show_high_Z, conf))

		# Mean absolute difference
		if MAD:
			_mad_(df,'SD')
			
		# Mean Square Error
		if MSE:
			mse = _mse_(df)
			print "\nMean Square Error = {0}".format(mse)

		# Plotting the expected frequncies (line) against the found ones(bars)
		if plot:
			_plot_dig_(df, x=x, y_Exp= df.Expected,y_Found=df.Found,\
			 N=N, figsize=(10,6), conf_Z=conf)

		if ret:
			return df

		### return df
	def first_digits(self, digs, inform=True, MAD=True, conf_level=95,\
		show_high_Z = 'pos', MSE=False, plot=True, ret = False):
		'''
		Performs the Benford First Digits test with the series of
		numbers provided, and populates the mapping dict for future
		selection of the original series.

		digs -> number of first digits to consider. Must be 1 (first digit),
			2 (first two digits) or 3 (first three digits).

		inform -> tells the number of registries that are being subjected to
			the Analysis; defaults to True

		MAD -> calculates the Mean of the Absolute Differences between the found
			and the expected distributions; defaults to True.

		conf_level -> confidence level to draw lower and upper limits when
			plotting and to limit the mapping of the proportions to only the
			ones significantly diverging from the expected. Defaults to 95.
			If None, no boundaries will be drawn.

		show_high_Z -> chooses which Z scores to be used when displaying results,
			according to the confidence level chosen. Defaluts to 'pos', which will
			highlight only the values that are higher than the expexted frequencies;
			'all' will highlight both found extremes (positive and negative); and
			an integer, which will use the first n entries, positive and negative,
			regardless of whether the Z is higher than the conf_level Z or not. 

		MSE -> calculates the Mean Square Error of the sample; defaults to False.

		plot -> draws the test plot for visual comparison, with the found
			distributions in bars and the expected ones in a line.

		
		'''
		N = len(self)

		if str(conf_level) not in self.confs.keys():
			raise ValueError("Value of parameter -conf_level- must be one\
 of the following: {0}".format(self.confs.keys()))

		if not digs in [1,2,3]:
			raise ValueError("The value assigned to the parameter -digs-\
 was {0}. Value must be 1, 2 or 3.".format(digs))

	# 	if not show_high_Z in ['pos', 'all'] or not isinstance(show_high_Z, int):
	# 		raise ValueError("The value of -show_high_Z- must be one of\
 # the following?: 'pos', 'all' or some integer.")
		
		# if digs == 1:
		# 	show_high_Z = 9

 		dig_name = 'F{0}D'.format(digs)
 		n,m = 10**(digs-1), 10**(digs)
		x = np.arange(n,m)
		conf = self.confs[str(conf_level)]
		
		if inform:
			print "\n---Test performed on {0} registries.---\n".format(N)
		# get the number of occurrences of the first two digits
		v = self[dig_name].value_counts()
		# get their relative frequencies
		p = self[dig_name].value_counts(normalize =True)
		# crate dataframe from them
		df = pd.DataFrame({'Counts': v, 'Found': p}).sort_index()
		# reindex from n to m in the case one or more of the first
		# digits are missing, so the Expected frequencies column
		# can later be joined; and swap NANs with zeros.
		if len(df.index) < m-n:
			df = df.reindex(x).fillna(0)
		# join the dataframe with the one of expected Benford's frequencies
		df = First(digs=digs,plot=False).join(df)
		# create column with absolute differences
		df['Dif'] = df.Found - df.Expected
		df['AbsDif'] = np.absolute(df.Dif)
		# calculate the Z-test column an display the dataframe by descending
		# Z test
		df['Z_test'] = _Z_test(df,N)
		#Populate dict with the most relevant entries
		self.maps[dig_name] = np.array(_inform_and_map_(df, inform,\
		 show_high_Z, conf))

		# Mean absolute difference
		if MAD:
			_mad_(df, test = dig_name)

		# Mean Square Error
		if MSE:
			mse = (df.AbsDif**2).mean()
			print "\nMean Square Error = {0}".format(mse)
		# Plotting the expected frequncies (line) against the found ones(bars)
		if plot:
			_plot_dig_(df, x = x, y_Exp = df.Expected, y_Found = df.Found,\
			 N = N, figsize = (5*(digs+1),4*(digs+.6)), conf_Z = conf)
		if ret:
			return df

		#return df
	def last_two_digits(self, inform=True, MAD=False, conf_level=95,\
	 	show_high_Z = 'pos', MSE=False, plot=True, ret = False):
		'''
		Performs the Benford Last Two Digits test with the series of
		numbers provided.

		inform -> tells the number of registries that are being subjected to
			the Analysis; defaults to True

		MAD -> calculates the Mean of the Absolute Differences between the found
			and the expected distributions; defaults to True.

		conf_level -> confidence level to draw lower and upper limits when
			plotting and to limit the mapping of the proportions to only the
			ones significantly diverging from the expected. Defaults to 95.
			If None, no boundaries will be drawn.

		show_high_Z -> chooses which Z scores to be used when displaying results,
			according to the confidence level chosen. Defaluts to 'pos', which will
			highlight only the values that are higher than the expexted frequencies;
			'all' will highlight both found extremes (positive and negative); and
			an integer, which will use the first n entries, positive and negative,
			regardless of whether the Z is higher than the conf_level Z or not

		MSE -> calculates the Mean Square Error of the sample; defaluts to False.

		plot -> draws the test plot for visual comparison, with the found
			distributions in bars and the expected ones in a line.

		'''
		if str(conf_level) not in self.confs.keys():
			raise ValueError("Value of -conf_level- must be one of the\
 following: {0}".format(self.confs.keys()))
		conf = self.confs[str(conf_level)]
		N = len(self)
		x = np.arange(0,100)
		if inform:
			print "\n---Test performed on " + str(N) + " registries.---\n"
		# get the number of occurrences of the last two digits
		v = self.L2D.value_counts()
		# get their relative frequencies
		p = self.L2D.value_counts(normalize =True)
		# crate dataframe from them
		df = pd.DataFrame({'Counts': v, 'Found': p}).sort_index()
		# join the dataframe with the one of expected Benford's frequencies
		df = LastTwo(plot=False).join(df)
		# create column with absolute differences
		df['Dif'] = df.Found - df.Expected
		df['AbsDif'] = np.absolute(df.Dif)
		# calculate the Z-test column an display the dataframe by descending
		# Z test
		df['Z_test'] = _Z_test(df,N)
		
		#Populate dict with the most relevant entries
		self.maps['L2D'] = np.array(_inform_and_map_(df, inform,\
		 show_high_Z, conf)).astype(int)

		# Mean absolute difference
		if MAD:
			_mad_(df, test='L2D')

		# Mean Square Error
		if MSE:
			mse = _mse_(df)
			print "\nMean Square Error = {0}".format(mse)
		# Plotting the expected frequencies (line) against the found ones (bars)
		if plot:
			_plot_dig_(df, x = x, y_Exp = df.Expected, y_Found =df.Found,\
			 N=N, figsize=(15,8), conf_Z=conf, text_x=True)
		
		if ret:
			return df
		
	
	def summation(self, digs=2, top=20, inform=True, plot=True, ret=False):
		'''
		Performs the Summation test. In a Benford series, the sums of the entries
		begining with the same digits tends to be the same.

		digs -> tells the first digits to use. 1- first; 2- first two;
				3- first three. Defaults to 2.

		top -> choses how many top values to show. Defaults to 20.

		plot -> plots the results. Defaults to True.
		'''

		if not digs in [1,2,3]:
			raise ValueError("The value assigned to the parameter -digs-\
 was {0}. Value must be 1, 2 or 3.".format(digs))
		#Set the future dict key
		if inform:
			N = len(self)
			print "\n---Test performed on {0} registries.---\n".format(N)
		dig_name = 'SUM{0}'.format(digs)
		if digs == 1:
			top = 9
		#Call the dict for F1D, F2D, F3D
		d = self.digs_dict[str(digs)]
		#Call the expected proportion according to digs
		l = 1./(9*(10**(digs-1)))

		s = self.groupby(d).sum()
		s['Percent'] = s.Seq.value_counts(normalize=True)
		s.columns.values[0] = 'Sum'
		s = s[['Sum','Percent']]
		s['AbsDif'] = np.absolute(s.Percent-l)

		#Populate dict with the most relevant entries
		self.maps[dig_name] = np.array(_inform_and_map_(s, inform,\
		 show_high_Z=top, conf=None)).astype(int)

		if plot:
			f = {'1':(8,5), '2':(13,8), '3':(21,13)}
			_plot_sum_(s, figsize=f[str(digs)], l=l)
		if ret:
			return s


	# def duplicates(self, inform=True, top_Rep=20):
		'''
		'''

		



def _Z_test(frame,N):
	'''
	Return the Z statistics for the proportions assessed

	frame -> DataFrame with the expected proportions and the already calculated
			Absolute Diferences between the found and expeccted proportions
	N -> sample size
	'''
	return (frame.AbsDif - (1/2*N))/(np.sqrt(frame.Expected*\
		(1-frame.Expected)/N))

def _mad_(frame, test):
	'''
	Returns the Mean Absolute Deviation (MAD) of the found proportions from the
	expected proportions. Then, it compares the found MAD with the accepted ranges 
	of the respective test.

	frame -> DataFrame with the Absolute Deviations already calculated.
	test -> Teste applied (F1D, SD, F2D...)
	'''
	mad = frame.AbsDif.mean()
	if test[0] == 'F':
		if test == 'F1D':
			margins = ['0.0006','0.0012','0.0015', 'Digit']
		elif test == 'F2D':
			margins = ['0.0012','0.0018','0.0022', 'Two Digits']
		else:
			margins = ['0.00036','0.00044','0.00050', 'Three Digits']
		print "\nThe Mean Absolute Deviation is {0}\n\
	For the First {1}:\n\
	- 0.0000 to {2}: Close Conformity\n\
	- {2} to {3}: Acceptable Conformity\n\
	- {3} to {4}: Marginally Acceptable Conformity\n\
	- Above {4}: Nonconformity".format(mad, margins[3], margins[0],\
	 margins[1], margins[2])

	elif test == 'SD':
		print "\nThe Mean Absolute Deviation is {0}\n\
	For the Second Digits:\n\
	- 0.0000 to 0.0008: Close Conformity\n\
	- 0.0008 to 0.0010: Acceptable Conformity\n\
	- 0.0010 to 0.0012: Marginally Acceptable Conformity\n\
	- Above 0.0012: Nonconformity".format(mad)

	else:
		print "\nThe Mean Absolute Deviation is {0}.\n".format(mad)

def _mse_(frame):
	'''
	Returns the test's Mean Square Error

	frane -> DataFrame with the already computed Absolute Deviations between
			the found and expected proportions
	'''
	return (frame.AbsDif**2).mean()

def _getMantissas_(arr):
	'''
	The mantissa is the non-integer part of the log of a number.
	This fuction uses the element-wise array operations of numpy
	to get the mantissas of each number's log.

	arr: np.array of integers or floats ---> np.array of floats
	'''
	log_a = np.log10(arr)
	return np.abs(log_a) - log_a.astype(int) # the number - its integer part


def _getMantissas2_(arr):
	'''
	The mantissa is the non-integer part of the log of a number.
	This fuction uses the element-wise array operations of numpy
	to get the mantissas of each number's log.

	arr: np.array of integers or floats ---> np.array of floats
	'''

	return np.abs(np.log10(arr)) % 1.0

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
	'''
	From a str sequence, return the characters that represent numbers

	seq -> string sequence
	'''
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

def _plot_dig_(df, x, y_Exp, y_Found, N, figsize, conf_Z, text_x=False):		
	'''
	Plots the digits tests results

	df -> DataFrame with the data to be plotted
	x -> sequence to be used in the x axis
	y_Exp -> sequence of the expected proportions to be used in the y axis (line)
	y_Found -> sequence of the found proportions to be used in the y axis (bars)
	N -> lenght of sequence, to be used when plotting the confidence levels
	figsize - > tuple to state the size of the plot figure
	conf_Z -> Confidence level
	text_x -> Forces to show all x ticks labels. Defaluts to True.
	'''

	fig = plt.figure(figsize=figsize)
	ax = fig.add_subplot(111)
	# plt.title('')
	# plt.xlabel('')
	# plt.ylabel(' (%)')
	ax.bar(x, y_Found * 100., label='Encontrada')
	ax.plot(x, y_Exp * 100., color='g',linewidth=2.5, label='Benford')
	ax.legend()
	if text_x:
		plt.xticks(x,df.index, rotation='vertical')
	# Plotting the Upper and Lower bounds considering the Z for the
	# informed confidence level
	if conf_Z != None:
		sig = conf_Z * np.sqrt(y_Exp*(1-y_Exp)/N) 
		upper = y_Exp + sig + (1/(2*N))
		lower = y_Exp - sig - (1/(2*N))
		ax.plot(x, upper * 100., color= 'r')
		ax.plot(x, lower * 100., color= 'r')
		ax.fill_between(x, upper * 100.,lower * 100., color='r', alpha=.3)
	plt.show()


def _plot_sum_(df, figsize, l):
	'''
	Plotss the summation test results

	df -> DataFrame with the data to be plotted
	figsize - > tuple to state the size of the plot figure
	l -> values with which to draw the horizontal line
	'''
	fig = plt.figure(figsize=figsize)
	ax = fig.add_subplot(111)
	plt.title('Expected vs. Found Sums')
	plt.xlabel('Digits')
	plt.ylabel('Sums')
	ax.bar(df.index, df.Percent, label='Found Sums')
	ax.axhline(l, color='r', linewidth=2)
	ax.legend()


def _collapse_scalar_(num, orders=2, dec=2):
	'''
	Collapses any number to a form defined by the user, with the chosen
	number of digits at the left of the floating point, with the chosen
	number of decimal digits or an int.
	num -> number to be collapsed
	orders -> orders of magnitude chosen (how many digts to the left of
		the floating point). Defaults to 2.
	dec -> number of decimal places. Defaults to 2. If 0 is chosen, returns
		an int.
	'''
	# Set n to 1 if the number is less than 1, since when num is less than 1
	# 10 must be raised to a smaller power   
	if num < 1:
		n = 1
    # Set n to 2 otherwise
	else:
		n = 2
    # Set the dividend l, which is 10 raised to the
    # integer part of the number's log
	l = 10 ** int(np.log10(num))
    # If dec is different than 0, use dec to round to the decimal
    # places chosen
	if dec != 0:
		return round(10. ** (orders + 1 - n) * num / l, dec)
    # If dec == 0, return integer
	else:
		return int(10. ** (orders + 1 - n) * num / l)


def _collapse_array_(arr, orders=2, dec=2):
	'''
	Collapses an array of numbers, each to a form defined by the user,
	with the chosen	number of digits at the left of the floating point,
	with the chosen	number of decimal digits or as ints.
	arr -> array of numbers to be collapsed
	orders -> orders of magnitude chosen (how many digts to the left of
		the floating point). Defaults to 2.
	dec -> number of decimal places. Defaults to 2. If 0 is chosen, returns
		array of integers.
	'''	
	# Create a array of ones with the lenght of the array to be collapsed,
	# for numberss less than 1, since when the number is less than 1
	# 10 must be raised to a smaller power 
	n = np.ones(len(arr))
    # Set the ones to two in the places where the array numbers are greater
    # or equal to one 
	n[arr>=1]=2
	# Set the dividend array l, composed of numbers 10 raised to the
    # integer part of the numbers' logs
	l = 10. ** (np.log10(arr).astype(int, copy=False))
	# If dec is different than 0, use dec to round to the decimal
    # places chosen
	if dec != 0:
		return 10. ** (orders + 1 - n) * arr / l
    # If dec == 0, return array of integers
	else:
		return (10. ** (orders + 1 - n) * arr / l).astype(int)


def _sanitize_float_(s, dec):
	s = str(s)
	if '.' in s or ',' in s:
		s = filter(type(s).isdigit, s)
		s = s[:-dec]+'.'+s[-dec:]
		return float(s)
	else:
		if filter(type(s).isdigit, s) == '':
			return np.nan
		else:
			return int(s)

def _sanitize_latin_float_(s, dec=2):
	s = str(s)
	s = filter(type(s).isdigit, s)
	return s[:-dec]+'.'+s[-dec:]

def _sanitize_latin_int_(s):
	s = str(s)
	s = filter(type(s).isdigit, s)
	return s

def _inform_and_map_(df, inform, show_high_Z, conf):
	'''
	
	'''

	if inform:
		if isinstance(show_high_Z, int):
			if conf != None:
				dd = df[['Expected','Found','Z_test']].sort_values('Z_test',\
				ascending=False).head(show_high_Z)
				print '\nThe entries with the top %s Z scores are:\n' % show_high_Z
			# Summation Test
			else:
				dd = df.sort_values('AbsDif', ascending=False).head(show_high_Z)
				print '\nThe entries with the top %s absolute deviations are:\n' % show_high_Z
		else:
			if show_high_Z == 'pos':
				m1 = df.Dif > 0
				m2 = df.Z_test > conf
				dd = df[['Expected','Found','Z_test']][m1 & m2].sort_values('Z_test',\
				 ascending=False)
				print '\nThe entries with the significant positive deviations are:\n'
			elif show_high_Z == 'neg':
				m1 = df.Dif < 0
				m2 = df.Z_test > conf
				dd = df[['Expected','Found','Z_test']][m1 & m2].sort_values('Z_test',\
				 ascending=False)
				print '\nThe entries with the significant negative deviations are:\n'
			else:
				dd = df[['Expected','Found','Z_test']][df.Z_test > conf].sort_values('Z_test',\
				 ascending=False)
				print '\nThe entries with the significant deviations are:\n'
		print dd
		return dd.index
	else:
		if isinstance(show_high_Z, int):
			if conf != None:
				dd = df[['Expected','Found','Z_test']].sort_values('Z_test',\
				ascending=False).head(show_high_Z)
			#Summation Test
			else:
				dd = df.sort_values('AbsDif', ascending=False).head(show_high_Z)
		else:
			if show_high_Z == 'pos':
				dd = df[['Expected','Found','Z_test']][df.Dif > 0 and \
				 df.Z_test > conf].sort_values('Z_test',ascending=False)
			elif show_high_Z == 'neg':
				dd = df[['Expected','Found','Z_test']][df.Dif < 0 and \
				 df.Z_test > conf].sort_values('Z_test',ascending=False)
			else:
				dd = df[['Expected','Found','Z_test']][df.Z_test > \
				conf].sort_values('Z_test', ascending=False)
		return dd.index

