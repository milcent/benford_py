'''
Benford_py for Python is a module for application of Benford's Law
to a sequence of numbers.

Dependent on pandas, numpy and matplotlib

All logarithms ar in base 10: "np.log10"

Copyright (C) 2014  Marcel Milcent

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.text import Annotation
import warnings
warnings.filterwarnings("default", category=PendingDeprecationWarning)


digs_dict = {1: 'F1D', 2: 'F2D', 3: 'F3D', 22: 'SD', -2: 'L2D'}

sec_order_dict = {key: f'{val}_sec' for key, val in digs_dict.items()}

rev_digs = {'F1D': 1, 'F2D': 2, 'F3D': 3, 'SD': 22, 'L2D': -2}

names = {'F1D': 'First Digit Test', 'F2D': 'First Two Digits Test',
         'F3D': 'First Three Digits Test', 'SD': 'Second Digit Test',
         'L2D': 'Last Two Digits Test',
         'F1D_sec': 'First Digit Second Order Test',
         'F2D_sec': 'First Two Digits Second Order Test',
         'F3D_sec': 'First Three Digits Second Order Test',
         'SD_sec': 'Second Digit Second Order Test',
         'L2D_sec': 'Last Two Digits Second Order Test',
         'F1D_Summ': 'First Digit Summation Test',
         'F2D_Summ': 'First Two Digits Summation Test',
         'F3D_Summ': 'First Three Digits Summation Test',
         'Mantissas': 'Mantissas Test'
         }

# Critical values for Mean Absolute Deviation
mad_dict = {1: [0.006, 0.012, 0.015], 2: [0.0012, 0.0018, 0.0022],
            3: [0.00036, 0.00044, 0.00050], 22: [0.008, 0.01, 0.012],
            -2: None, 'F1D': 'First Digit', 'F2D': 'First Two Digits',
            'F3D': 'First Three Digits', 'SD': 'Second Digits'}

# Color for the plotting
colors = {'m': '#00798c', 'b': '#E2DCD8', 's': '#9c3848',
          'af': '#edae49', 'ab': '#33658a', 'h': '#d1495b',
          'h2': '#f64740', 't': '#16DB93'}

# Critical Z-scores according to the confindence levels
confs = {None: None, 80: 1.285, 85: 1.435, 90: 1.645, 95: 1.96,
         99: 2.576, 99.9: 3.29, 99.99: 3.89, 99.999: 4.417,
         99.9999: 4.892, 99.99999: 5.327}

p_values = {None: 'None', 80: '0.2', 85: '0.15', 90: '0.1', 95: '0.05',
            99: '0.01', 99.9: '0.001', 99.99: '0.0001', 99.999: '0.00001',
            99.9999: '0.000001', 99.99999: '0.0000001'}

# Critical Chi-Square values according to the tests degrees of freedom
# and confidence levels
crit_chi2 = {8: {80: 11.03, 85: 12.027, 90: 13.362, 95: 15.507,
                 99: 20.090, 99.9: 26.124, 99.99: 31.827, None: None,
                 99.999: 37.332, 99.9999: 42.701, 99.99999: 47.972},
             9: {80: 12.242, 85: 13.288, 90: 14.684, 95: 16.919,
                 99: 21.666, 99.9: 27.877, 99.99: 33.72, None: None,
                 99.999: 39.341, 99.9999: 44.811, 99.99999: 50.172},
             89: {80: 99.991, 85: 102.826, 90: 106.469, 95: 112.022,
                  99: 122.942, 99.9: 135.978, 99.99: 147.350,
                  99.999: 157.702, 99.9999: 167.348, 99.99999: 176.471,
                  None: None},
             99: {80: 110.607, 85: 113.585, 90: 117.407,
                  95: 123.225, 99: 134.642, 99.9: 148.230,
                  99.99: 160.056, 99.999: 170.798, 99.9999: 180.792,
                  99.99999: 190.23, None: None},
             899: {80: 934.479, 85: 942.981, 90: 953.752, 95: 969.865,
                   99: 1000.575, 99.9: 1035.753, 99.99: 1065.314,
                   99.999: 1091.422, 99.9999: 1115.141,
                   99.99999: 1137.082, None: None}
             }

# Critical Kolmogorov-Smirnoff values according to the confidence levels 
KS_crit = {80: 1.075, 85: 1.139, 90: 1.125, 95: 1.36, 99: 1.63,
           99.9: 1.95, 99.99: 2.23, 99.999: 2.47,
           99.9999: 2.7, 99.99999: 2.9, None: None}


class First(pd.DataFrame):
    '''
     Returns the expected probabilities of the First, First Two, or
     First Three digits according to Benford's distribution.

    Parameters
    ----------

    digs-> 1, 2 or 3 - tells which of the first digits to consider:
            1 for the First Digit, 2 for the First Two Digits and 3 for
            the First Three Digits.

    plot-> option to plot a bar chart of the Expected proportions.
            Defaults to True.
    '''

    def __init__(self, digs, plot=True):
        _check_digs_(digs)
        dig_name = f'First_{digs}_Dig'
        Dig = np.arange(10 ** (digs - 1), 10 ** digs)
        Exp = np.log10(1 + (1. / Dig))

        pd.DataFrame.__init__(self, {'Expected': Exp}, index=Dig)
        self.index.names = [dig_name]

        if plot:
            _plot_expected_(self, digs)


class Second(pd.DataFrame):
    '''
    Returns the expected probabilities of the Second Digits
    according to Benford's distribution.

    Parameters
    ----------

    plot: option to plot a bar chart of the Expected proportions.
        Defaults to True.
    '''
    def __init__(self, plot=True):
        a = np.arange(10, 100)
        Expe = np.log10(1 + (1. / a))
        Sec_Dig = np.array(list(range(10)) * 9)

        df = pd.DataFrame({'Expected': Expe, 'Sec_Dig': Sec_Dig})

        pd.DataFrame.__init__(self, df.groupby('Sec_Dig').sum())

        if plot:
            _plot_expected_(self, 22)


class LastTwo(pd.DataFrame):
    '''
    Returns the expected probabilities of the Last Two Digits
    according to Benford's distribution.

    Parameters
    ----------

    plot: option to plot a bar chart of the Expected proportions.
        Defaults to True.
    '''
    def __init__(self, num=False, plot=True):
        exp = np.array([1 / 99.] * 100)
        pd.DataFrame.__init__(self, {'Expected': exp,
                              'Last_2_Dig': _lt_(num=num)})
        self.set_index('Last_2_Dig', inplace=True)
        if plot:
            _plot_expected_(self, -2)


class Base(pd.DataFrame):
    '''
    Internalizes and prepares the data for Analysis.

    Parameters
    ----------

    data: sequence of numbers to be evaluated. Must be a numpy 1D array,
        a pandas Series or a pandas DataFrame column, with values being
        integers or floats.

    decimals: number of decimal places to consider. Defaluts to 2.
        If integers, set to 0. If set to -infer-, it will remove the zeros
        and consider up to the fifth decimal place to the right, but will
        loose performance.

    sign: tells which portion of the data to consider. pos: only the positive
        entries; neg: only negative entries; all: all entries but zeros.
        Defaults to all.`
    '''
    def __init__(self, data, decimals, sign='all', sec_order=False):

        pd.DataFrame.__init__(self, {'Seq': data})

        if (self.Seq.dtypes != 'float64') & (self.Seq.dtypes != 'int64'):
            raise TypeError("The sequence dtype was not pandas int64 nor "
                            "float64. Convert it to whether int of float, "
                            "and try again.")

        if sign == 'all':
            self.Seq = self.Seq.loc[self.Seq != 0]
        elif sign == 'pos':
            self.Seq = self.Seq.loc[self.Seq > 0]
        else:
            self.Seq = self.Seq.loc[self.Seq < 0]

        self.dropna(inplace=True)

        ab = self.Seq.abs()

        if self.Seq.dtypes == 'int64':
            self['ZN'] = ab
        else:
            if decimals == 'infer':
                self['ZN'] = ab.astype(str).str\
                               .replace('.', '')\
                               .str.lstrip('0')\
                               .str[:5].astype(int)
            else:
                self['ZN'] = (ab * (10 ** decimals)).astype(int)
        # First digits
        for col in ['F1D', 'F2D', 'F3D']:
            temp = self.ZN.loc[self.ZN >= 10 ** (rev_digs[col] - 1)]
            self[col] = (temp // 10 ** ((np.log10(temp).astype(int)) -
                                        (rev_digs[col] - 1)))
            # fill NANs with -1, which is a non-usable value for digits,
            # to be discarded later.
            self[col] = self[col].fillna(-1).astype(int)
        # Second digit
        temp_sd = self.loc[self.ZN >= 10]
        self['SD'] = (temp_sd.ZN // 10**((np.log10(temp_sd.ZN)).astype(int) -
                                         1)) % 10
        self['SD'] = self['SD'].fillna(-1).astype(int)
        # Last two digits
        temp_l2d = self.loc[self.ZN >= 1000]
        self['L2D'] = temp_l2d.ZN % 100
        self['L2D'] = self['L2D'].fillna(-1).astype(int)


class Test(pd.DataFrame):
    '''
    Transforms the original number sequence into a DataFrame reduced
    by the ocurrences of the chosen digits, creating other computed
    columns

    Parameters
    ----------

    base: The Base object with the data prepared for Analysis

    digs: Tells which test to perform -> 1: first digit; 2: first two digits;
        3: furst three digits; 22: second digit; -2: last two digits.
    
    confidence: confidence level to draw lower and upper limits when
        plotting and to limit the top deviations to show.

    limit_N: sets a limit to N as the sample size for the calculation of
            the Z scores if the sample is too big. Defaults to None.
    '''

    def __init__(self, base, digs, confidence, limit_N=None, sec_order=False):
        # create a separated Expected distributions object
        super(Test, self).__init__(_test_(digs))
        # create column with occurrences of the digits in the base
        self['Counts'] = base[digs_dict[digs]].value_counts()
        # create column with relative frequencies
        self['Found'] = base[digs_dict[digs]].value_counts(normalize=True)
        self.fillna(0, inplace=True)
        # create column with absolute differences
        self['Dif'] = self.Found - self.Expected
        self['AbsDif'] = np.absolute(self.Dif)
        self.N = _set_N_(len(base), limit_N)
        self['Z_score'] = _Z_score(self, self.N)
        self.chi_square = _chi_square_2(self)
        self.KS = _KS_2(self)
        self.MAD = self.AbsDif.mean()
        self.ddf = len(self) - 1
        self.confidence = confidence
        self.digs = digs
        self.sec_order = sec_order

        if sec_order:
            self.name = names[sec_order_dict[digs]]
        else:
            self.name = names[digs_dict[digs]]
    
    def update_confidence(self, new_conf, check=True):
        '''
        Sets a new confidence level for the Benford object, so as to be used to
        produce critical values for the tests

        Parameters
        ----------

        new_conf -> new confidence level to draw lower and upper limits when
            plotting and to limit the top deviations to show, as well as to
            calculate critical values for the tests' statistics.
        
        check -> checks the value provided for the confidence. Defaults to True
        '''
        if check:
            self.confidence = _check_confidence_(new_conf)
        else:
            self.confidence = new_conf

    @property
    def critical_values(self):
        '''
        Returns a dict with the critical values for the test at hand, accroding
        to the current confidence level.
        '''
        return {'Z': confs[self.confidence],
                'KS': KS_crit[self.confidence] / self.N ** 0.5,
                'chi2': crit_chi2[self.ddf][self.confidence],
                'MAD': mad_dict[self.digs]
                }

    def show_plot(self):
        '''
        Draws the test plot.
        '''
        x, figsize, text_x = _get_plot_args(self.digs)
        _plot_dig_(self, x=x, y_Exp=self.Expected, y_Found=self.Found,
                    N=self.N, figsize=figsize, conf_Z=confs[self.confidence],
                    text_x=text_x
                    )

    def report(self, high_Z='pos', show_plot=True):
        '''
        Handles the report especific to the test, considering its statistics
        and according to the current confidence level.

        Parameters
        ----------
        high_Z: chooses which Z scores to be used when displaying results,
            according to the confidence level chosen. Defaluts to 'pos',
            which will highlight only values higher than the expexted
            frequencies; 'all' will highlight both extremes (positive and
            negative); and an integer, which will use the first n entries,
            positive and negative, regardless of whether Z is higher than
            the critical value or not.
        show_plot: calls the show_plot method, to draw the test plot
        '''
        high_Z = _check_high_Z_(high_Z)
        _report_test_(self, high_Z, self.critical_values)
        if show_plot:
            self.show_plot()

class Summ(pd.DataFrame):
    '''
    Gets the base object and outputs a Summation test object

    Parameters
    ----------
    base: The Base object with the data prepared for Analysis

    test: The test for which to compute the summation

    '''
    def __init__(self, base, test):
        super(Summ, self).__init__(base.abs()
                                   .groupby(test)[['Seq']]
                                   .sum())
        self['Percent'] = self.Seq / self.Seq.sum()
        self.columns.values[0] = 'Sum'
        self.expected = 1 / len(self)
        self['AbsDif'] = np.absolute(self.Percent - self.expected)
        self.index = self.index.astype(int)
        self.MAD = self.AbsDif.mean()
        self.confidence = None
        self.digs = rev_digs[test]
        self.name = names[f'{test}_Summ']

    def show_plot(self):
        '''
        Draws the Summation test plot.
        '''
        figsize=(2 * (self.digs ** 2 + 5), 1.5 * (self.digs ** 2 + 5))
        _plot_sum_(self, figsize, self.expected)
    
    def report(self, high_diff=None, show_plot=True):
        '''
        Gives the report on the Summation test.
        -----------
        Parameters

        high_diff: Number of records to show after ordering by the absolute
            differences between the found and the expected proportions
        
        show_plot: calls the show_plot method, to draw the Summation test plot
        '''
        _report_test_(self, high_diff)
        if show_plot:
            self.show_plot()
        
class Benford(object):
    '''
    Initializes a Benford Analysis object and computes the proportions for
    the digits. The tets dataFrames are atributes, i.e., obj.F1D is the First
    Digit DataFrame, the obj.F2D,the First Two Digits one, and so one, F3D for
    First Three Digits, SD for Second  Digit and L2D for Last Two Digits.

    Parameters
    ----------
    data: sequence of numbers to be evaluated. Must be a numpy 1D array,
        a pandas Series or a tuple with a pandas DataFrame and the name (str)
        of the chosen column. Values must be integers or floats.

    decimals: number of decimal places to consider. Defaluts to 2.
        If integers, set to 0. If set to -infer-, it will remove the zeros
        and consider up to the fifth decimal place to the right, but will
        loose performance.

    sign: tells which portion of the data to consider. pos: only the positive
        entries; neg: only negative entries; all: all entries but zeros.
        Defaults to all.
    
    confidence: confidence level to draw lower and upper limits when
            plotting and to limit the top deviations to show, as well as to
            calculate critical values for the tests' statistics. Defaults to 95.

    sec_order: runs the Second Order tests, which are the Benford's tests
        performed on the differences between the ordered sample (a value minus
        the one before it, and so on). If the original series is Benford-
        compliant, this new sequence should aldo follow Beford. The Second
        Order can also be called separately, through the method sec_order().

    summation: creates the Summation DataFrames for the First, First Two, and
        First Three Digits. The summation tests can also be called separately,
        through the method summation().

    limit_N: sets a limit to N as the sample size for the calculation of
        the Z scores if the sample is too big. Defaults to None.

    verbose: gives some information about the data and the registries used
        and discarded for each test.
    '''

    def __init__(self, data, decimals=2, sign='all', confidence=95,
                 mantissas=False, sec_order=False, summation=False,
                 limit_N=None, verbose=True):
        self.data, self.chosen = _input_data_(data)
        self.decimals = decimals
        self.sign = sign
        self.confidence = _check_confidence_(confidence)
        self.limit_N = limit_N
        self.verbose = verbose
        self.base = Base(self.chosen, decimals, sign)
        self.tests = []

        # Create a DatFrame for each Test
        for key, val in digs_dict.items():
            test = Test(self.base.loc[self.base[val] != -1],
                        digs=key, confidence=self.confidence,
                        limit_N=self.limit_N)
            setattr(self, val, test)
            self.tests.append(val)
        # dict with the numbers of discarded entries for each test column
        self._discarded = {key: val for (key, val) in
                           zip(digs_dict.values(),
                               [len(self.base[col].loc[self.base[col] == -1])
                                for col in digs_dict.values()])}

        if self.verbose:
            print('\n',' Benford Object Instantiated '.center(50, '#'),'\n')
            print(f'Initial sample size: {len(self.chosen)}.\n')
            print(f'Test performed on {len(self.base)} registries.\n')
            print(f'Number of discarded entries for each test:\n{self._discarded}')

        if mantissas:
            self.mantissas()
    
        if sec_order:
            self.sec_order()

        if summation:
            self.summation()
    
    def update_confidence(self, new_conf, tests=None):
        '''
        Sets a new confidence level for the Benford object, so as to be used to
        produce critical values for the tests

        Parameters
        ----------
        new_conf -> new confidence level to draw lower and upper limits when
            plotting and to limit the top deviations to show, as well as to
            calculate critical values for the tests' statistics.
        tests -> list of tests names (strings) to have their confidence updated.
            If only one, provide a one-element list, like ['F1D']. Defauts to
            None, in which case it will use the instance .test list attribute.
        '''
        self.confidence = _check_confidence_(new_conf)
        if tests is None:
            tests = self.tests
        else:
            if not isinstance(tests, list):
                raise ValueError('tests must be a list or None.')
        for test in tests:
            try:
                getattr(self, test).update_confidence(self.confidence, check=False)
            except AttributeError:
                if test in ['Mantissas', 'F1D_Summ', 'F2D_Summ', 'F3D_Summ']:
                    pass
                else:
                    print(f"{test} not in Benford instance tests - review test's name.")
                    pass
    
    @property
    def all_confidences(self):
        '''
        Returns the confidence level for the instance's tests, when applicable 
        '''
        con_dic= {}
        for key in self.tests:
            try:
                con_dic[key] = getattr(self, key).confidence
            except AttributeError:
                pass
        return con_dic

    def mantissas(self):
        """ 
        Adds a Mantissas object to the tests, with all its statistics and
        plotting capabilities. 
        """
        self.Mantissas = Mantissas(self.base.Seq)
        self.tests.append('Mantissas')
        if self.verbose:
            print('\nAdded Mantissas test.')

    def sec_order(self):
        '''
        Runs the Second Order tests, which are the Benford's tests
        performed on the differences between the ordered sample (a value minus
        the one before it, and so on). If the original series is Benford-
        compliant, this new sequence should aldo follow Beford. The Second
        Order can also be called separately, through the method sec_order().
        '''
        self.base_sec = Base(_subtract_sorted_(self.chosen),
                             decimals=self.decimals, sign=self.sign)
        for key, val in digs_dict.items():
            test = Test(self.base_sec.loc[self.base_sec[val] != -1],
                        digs=key, confidence=self.confidence,
                        limit_N=self.limit_N, sec_order=True)
            setattr(self, sec_order_dict[key], test)
            self.tests.append(f'{val}_sec')
            # No need to populate crit_vals dict, since they are the
            # same and do not depend on N
            self._discarded_sec = {key: val for (key, val) in zip(
                                   sec_order_dict.values(),
                                   [sum(self.base_sec[col] == -1) for col in
                                    digs_dict.values()])}
        if self.verbose:
            print(f'\nSecond order tests run in {len(self.base_sec)} '
                  'registries.\n\nNumber of discarded entries for second order'
                  f' tests:\n{self._discarded_sec}')

    def summation(self):
        '''
        Creates Summation test DataFrames from Base object
        '''
        for test in ['F1D', 'F2D', 'F3D']:
            t =  f'{test}_Summ'
            setattr(self, t, Summ(self.base, test))
            self.tests.append(t)

        if self.verbose:
            print('\nAdded Summation DataFrames to F1D, F2D and F3D Tests.')


class Source(pd.DataFrame):
    '''
    Prepares the data for Analysis. pandas DataFrame subclass.

    Parameters
    ----------

    data: sequence of numbers to be evaluated. Must be a numpy 1D array,
        a pandas Series or a pandas DataFrame column, with values being
        integers or floats.

    decimals: number of decimal places to consider. Defaluts to 2.
        If integers, set to 0. If set to -infer-, it will remove the zeros
        and consider up to the fifth decimal place to the right, but will
        loose performance.

    sign: tells which portion of the data to consider. pos: only the positive
        entries; neg: only negative entries; all: all entries but zeros.
        Defaults to all.`

    sec_order: choice for the Second Order Test, which cumputes the
        differences between the ordered entries before running the Tests.

    verbose: tells the number of registries that are being subjected to
        the analysis; defaults to True
    '''

    def __init__(self, data, decimals=2, sign='all', sec_order=False,
                 verbose=True):

        if sign not in ['all', 'pos', 'neg']:
            raise ValueError("The -sign- argument must be "
                             "'all','pos' or 'neg'.")

        pd.DataFrame.__init__(self, {'Seq': data})

        if self.Seq.dtypes != 'float64' and self.Seq.dtypes != 'int64':
            raise TypeError('The sequence dtype was not pandas int64 nor float64.\n'
                            'Convert it to whether int64 of float64, and try again.')

        if sign == 'pos':
            self.Seq = self.Seq.loc[self.Seq > 0]
        elif sign == 'neg':
            self.Seq = self.Seq.loc[self.Seq < 0]
        else:
            self.Seq = self.Seq.loc[self.Seq != 0]

        self.dropna(inplace=True)

        if verbose:
            print(f"\nInitialized sequence with {len(self)} registries.")
        if sec_order:
            self.Seq = _subtract_sorted_(self.Seq.copy())
            self.dropna(inplace=True)
            self.reset_index(inplace=True)
            if verbose:
                print('Second Order Test. Initial series reduced '
                      f'to {len(self.Seq)} entries.')

        ab = self.Seq.abs()

        if self.Seq.dtypes == 'int64':
            self['ZN'] = ab
        else:
            if decimals == 'infer':
                # There is some numerical issue with Windows that required
                # implementing it differently (and slower)
                self['ZN'] = ab.astype(str)\
                               .str.replace('.', '')\
                               .str.lstrip('0').str[:5]\
                               .astype(int)
            else:
                self['ZN'] = (ab * (10 ** decimals)).astype(int)

    def mantissas(self, report=True, plot=True, figsize=(15, 8)):
        '''
        Calculates the mantissas, their mean and variance, and compares them
        with the mean and variance of a Benford's sequence.

        Parameters
        ----------
        report: prints the mamtissas mean, variance, skewness and kurtosis
            for the sequence studied, along with reference values.
        plot: plots the ordered mantissas and a line with the expected
            inclination. Defaults to True.

        figsize -> tuple that sets the figure size
        '''
        self['Mant'] = _getMantissas_(np.abs(self.Seq))
        if report:
            p = self[['Seq', 'Mant']]
            p = p.loc[p.Seq > 0].sort_values('Mant')
            print(f"The Mantissas MEAN is {p.Mant.mean()}. Ref: 0.5.")
            print(f"The Mantissas VARIANCE is {p.Mant.var()}. Ref: 0.083333.")
            print(f"The Mantissas SKEWNESS is {p.Mant.skew()}. \tRef: 0.")
            print(f"The Mantissas KURTOSIS is {p.Mant.kurt()}. \tRef: -1.2.")

        if plot:
            N = len(p)
            p['x'] = np.arange(1, N + 1)
            n = np.ones(N) / N
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            ax.plot(p.x, p.Mant, 'r-', p.x, n.cumsum(), 'b--',
                    linewidth=2)
            plt.ylim((0, 1.))
            plt.xlim((1, N + 1))
            plt.show()

    def first_digits(self, digs, verbose=True, confidence=None, high_Z='pos',
                     limit_N=None, MAD=False, MSE=False, chi_square=False,
                     KS=False, show_plot=True, simple=False, ret_df=False,
                     inform=None):
        '''
        Performs the Benford First Digits test with the series of
        numbers provided, and populates the mapping dict for future
        selection of the original series.

        Parameters
        ----------

        digs -> number of first digits to consider. Must be 1 (first digit),
            2 (first two digits) or 3 (first three digits).

        verbose: tells the number of registries that are being subjected to
            the analysis; defaults to True

        digs: number of first digits to consider. Must be 1 (first digit),
            2 (first two digits) or 3 (first three digits).

        confidence: confidence level to draw lower and upper limits when
            plotting and to limit the top deviations to show, as well as to
            calculate critical values for the tests' statistics. Defaults to None.

        high_Z: chooses which Z scores to be used when displaying results,
            according to the confidence level chosen. Defaluts to 'pos',
            which will highlight only values higher than the expexted
            frequencies; 'all' will highlight both extremes (positive and
            negative); and an integer, which will use the first n entries,
            positive and negative, regardless of whether Z is higher than
            the confidence or not.

        limit_N: sets a limit to N as the sample size for the calculation of
            the Z scores if the sample is too big. Defaults to None.

        MAD: calculates the Mean Absolute Difference between the
            found and the expected distributions; defaults to False.

        MSE: calculates the Mean Square Error of the sample; defaults to
            False.

        show_plot: draws the test plot. Defaults to True.

        ret_df: returns the test DataFrame. Defaults to False. True if run by
            the test function.
        '''
        # Check on the possible values for confidence levels
        confidence = _check_confidence_(confidence)
        # Check on possible digits
        _check_test_(digs)

        verbose = _deprecate_inform_(verbose, inform)

        temp = self.loc[self.ZN >= 10 ** (digs - 1)]
        temp[digs_dict[digs]] = (temp.ZN // 10 ** ((np.log10(temp.ZN).astype(
                                                   int)) - (digs - 1))).astype(
                                                       int)
        n, m = 10 ** (digs - 1), 10 ** (digs)
        x = np.arange(n, m)

        if simple:
            verbose = False
            show_plot = False
            df = _prep_(temp[digs_dict[digs]], digs, limit_N=limit_N,
                        simple=True, confidence=None)
        else:
            N, df = _prep_(temp[digs_dict[digs]], digs, limit_N=limit_N,
                           simple=False, confidence=confidence)

        if verbose:
            print(f"\nTest performed on {len(temp)} registries.\n"
                  f"Discarded {len(self) - len(temp)} records < {10 ** (digs - 1)}"
                  " after preparation.")
            if confidence is not None:
                _inform_(df, high_Z=high_Z, conf=confs[confidence])

        # Mean absolute difference
        if MAD:
            self.MAD = _mad_(df, test=digs, verbose=verbose)

        # Mean Square Error
        if MSE:
            self.MSE = _mse_(df, verbose=verbose)

        # Chi-square statistic
        if chi_square:
            self.chi_square = _chi_square_(df, ddf=len(df) - 1,
                                           confidence=confidence,
                                           verbose=verbose)
        # KS test
        if KS:
            self.KS = _KS_(df, confidence=confidence, N=len(temp),
                           verbose=verbose)

        # Plotting the expected frequncies (line) against the found ones(bars)
        if show_plot:
            _plot_dig_(df, x=x, y_Exp=df.Expected, y_Found=df.Found, N=N,
                       figsize=(2 * (digs ** 2 + 5), 1.5 * (digs ** 2 + 5)),
                       conf_Z=confs[confidence])
        if ret_df:
            return df

    def second_digit(self, verbose=True, confidence=None, high_Z='pos',
                     limit_N=None, MAD=False, MSE=False, chi_square=False,
                     KS=False, show_plot=True, simple=False, ret_df=False,
                     inform=None):
        '''
        Performs the Benford Second Digit test with the series of
        numbers provided.

        verbose-> tells the number of registries that are being subjected to
            the analysis; defaults to True

        MAD: calculates the Mean Absolute Difference between the
            found and the expected distributions; defaults to False.

        confidence: confidence level to draw lower and upper limits when
            plotting and to limit the top deviations to show, as well as to
            calculate critical values for the tests' statistics. Defaults to None.

        high_Z: chooses which Z scores to be used when displaying results,
            according to the confidence level chosen. Defaluts to 'pos',
            which will highlight only values higher than the expexted
            frequencies; 'all' will highlight both extremes (positive and
            negative); and an integer, which will use the first n entries,
            positive and negative, regardless of whether Z is higher than
            the confidence or not.

        limit_N: sets a limit to N as the sample size for the calculation of
            the Z scores if the sample is too big. Defaults to None.

        MSE: calculates the Mean Square Error of the sample; defaults to
            False.

        show_plot: draws the test plot.

        ret_df: returns the test DataFrame. Defaults to False. True if run by
            the test function.
        '''
        confidence = _check_confidence_(confidence)

        conf = confs[confidence]

        verbose = _deprecate_inform_(verbose, inform)

        temp = self.loc[self.ZN >= 10]
        temp['SD'] = (temp.ZN // 10**((np.log10(temp.ZN)).astype(
                      int) - 1)) % 10

        if simple:
            verbose = False
            show_plot = False
            df = _prep_(temp['SD'], 22, limit_N=limit_N, simple=True,
                        confidence=None)
        else:
            N, df = _prep_(temp['SD'], 22, limit_N=limit_N, simple=False,
                           confidence=confidence)

        if verbose:
            print(f"\nTest performed on {len(temp)} registries.\nDiscarded "
                  f"{len(self) - len(temp)} records < 10 after preparation.")
            if confidence is not None:
                _inform_(df, high_Z, conf)

        # Mean absolute difference
        if MAD:
            self.MAD = _mad_(df, test=22, verbose=verbose)

        # Mean Square Error
        if MSE:
            self.MSE = _mse_(df, verbose=verbose)

        # Chi-square statistic
        if chi_square:
            self.chi_square = _chi_square_(df, ddf=9, confidence=confidence,
                                           verbose=verbose)
        # KS test
        if KS:
            self.KS = _KS_(df, confidence=confidence, N=len(temp),
                           verbose=verbose)

        # Plotting the expected frequncies (line) against the found ones(bars)
        if show_plot:
            _plot_dig_(df, x=np.arange(0, 10), y_Exp=df.Expected,
                       y_Found=df.Found, N=N, figsize=(10, 6), conf_Z=conf)
        if ret_df:
            return df

    def last_two_digits(self, verbose=True, confidence=None, high_Z='pos',
                        limit_N=None, MAD=False, MSE=False, chi_square=False,
                        KS=False, show_plot=True, simple=False, ret_df=False,
                        inform=None):
        '''
        Performs the Benford Last Two Digits test with the series of
        numbers provided.

        verbose-> tells the number of registries that are being subjected to
            the analysis; defaults to True

        MAD: calculates the Mean Absolute Difference between the
            found and the expected distributions; defaults to False.

        confidence: confidence level to draw lower and upper limits when
            plotting and to limit the top deviations to show, as well as to
            calculate critical values for the tests' statistics. Defaults to None.

        high_Z: chooses which Z scores to be used when displaying results,
            according to the confidence level chosen. Defaluts to 'pos',
            which will highlight only values higher than the expexted
            frequencies; 'all' will highlight both extremes (positive and
            negative); and an integer, which will use the first n entries,
            positive and negative, regardless of whether Z is higher than
            the confidence or not.

        limit_N: sets a limit to N as the sample size for the calculation of
            the Z scores if the sample is too big. Defaults to None.

        MSE: calculates the Mean Square Error of the sample; defaults to
            False.

        show_plot: draws the test plot.

        '''
        confidence = _check_confidence_(confidence)
        conf = confs[confidence]

        verbose = _deprecate_inform_(verbose, inform)

        temp = self.loc[self.ZN >= 1000]
        temp['L2D'] = temp.ZN % 100

        if simple:
            verbose = False
            show_plot = False
            df = _prep_(temp['L2D'], -2, limit_N=limit_N, simple=True,
                        confidence=None)
        else:
            N, df = _prep_(temp['L2D'], -2, limit_N=limit_N, simple=False,
                           confidence=confidence)

        if verbose:
            print(f"\nTest performed on {len(temp)} registries.\n\nDiscarded "
                  f"{len(self) - len(temp)} records < 1000 after preparation")
            if confidence is not None:
                _inform_(df, high_Z, conf)

        # Mean absolute difference
        if MAD:
            self.MAD = _mad_(df, test=-2, verbose=verbose)

        # Mean Square Error
        if MSE:
            self.MSE = _mse_(df, verbose=verbose)

        # Chi-square statistic
        if chi_square:
            self.chi_square = _chi_square_(df, ddf=99, confidence=confidence,
                                           verbose=verbose)
        # KS test
        if KS:
            self.KS = _KS_(df, confidence=confidence, N=len(temp),
                           verbose=verbose)

        # Plotting expected frequencies (line) versus found ones (bars)
        if show_plot:
            _plot_dig_(df, x=np.arange(0, 100), y_Exp=df.Expected,
                       y_Found=df.Found, N=N, figsize=(15, 5),
                       conf_Z=conf, text_x=True)
        if ret_df:
            return df

    def summation(self, digs=2, top=20, verbose=True, show_plot=True,
                  ret_df=False, inform=None):
        '''
        Performs the Summation test. In a Benford series, the sums of the
        entries begining with the same digits tends to be the same.

        digs -> tells the first digits to use. 1- first; 2- first two;
                3- first three. Defaults to 2.

        top -> choses how many top values to show. Defaults to 20.

        show_plot -> plots the results. Defaults to True.
        '''
        _check_digs_(digs)

        verbose = _deprecate_inform_(verbose, inform)

        if digs == 1:
            top = 9
        # Call the dict for F1D, F2D, F3D
        d = digs_dict[digs]
        if d not in self.columns:
            self[d] = self.ZN.astype(str).str[:digs].astype(int)
        # Call the expected proportion according to digs
        li = 1. / (9 * (10 ** (digs - 1)))

        df = self.groupby(d).sum()
        # s.drop(0, inplace=True)
        df['Percent'] = df.ZN / df.ZN.sum()
        df.columns.values[1] = 'Summ'
        df = df[['Sum', 'Percent']]
        df['AbsDif'] = np.absolute(df.Percent - li)

        # Populate dict with the most relevant entries
        # self.maps[dig_name] = np.array(_inform_and_map_(s, inform,
        #                                high_Z=top, conf=None)).astype(int)
        if verbose:
            # N = len(self)
            print(f"\nTest performed on {len(self)} registries.\n")
            print(f"The top {top} diferences are:\n")
            print(df[:top])

        if show_plot:
            _plot_sum_(df, figsize=(
                       2 * (digs ** 2 + 5), 1.5 * (digs ** 2 + 5)), li=li)

        if ret_df:
            return df

    def duplicates(self, verbose=True, top_Rep=20, inform=None):
        '''
        Performs a duplicates test and maps the duplicates count in descending
        order.

        verbose -> tells how many duplicated entries were found and prints the
            top numbers according to the top_Rep parameter. Defaluts to True.

        top_Rep -> int or None. Chooses how many duplicated entries will be
            shown withe the top repititions. Defaluts to 20. If None, returns
            al the ordered repetitions.
        '''
        if top_Rep is not None and not isinstance(top_Rep, int):
            raise ValueError('The top_Rep parameter must be an int or None.')

        verbose = _deprecate_inform_(verbose, inform)

        dup = self[['Seq']][self.Seq.duplicated(keep=False)]
        dup_count = dup.groupby(self.Seq).count()

        dup_count.index.names = ['Entries']
        dup_count.rename(columns={'Seq': 'Count'}, inplace=True)

        dup_count.sort_values('Count', ascending=False, inplace=True)

        self.maps['dup'] = dup_count.index[:top_Rep].values  # np.array

        if verbose:
            print(f'\nFound {len(dup_count)} duplicated entries.\n'
                  f'The entries with the {top_Rep} highest repitition counts are:')
            print(dup_count.head(top_Rep))
        else:
            return dup_count(top_Rep)


class Mantissas(object):
    '''
    Returns a Series with the data mantissas,

    Parameters
    ----------
    data: sequence to compute mantissas from, numpy 1D array, pandas
        Series of pandas DataFrame column.
    '''

    def __init__(self, data):

        data = pd.Series(_check_num_array(data))
        data = data.dropna().loc[data != 0].abs()
        
        self.data = pd.DataFrame({'Mantissa': _getMantissas_(np.abs(data))})

        self.stats = {'Mean': self.data.Mantissa.mean(),
                      'Var': self.data.Mantissa.var(),
                      'Skew': self.data.Mantissa.skew(),
                      'Kurt': self.data.Mantissa.kurt()}

    def verbose(self):
        '''
        Shows the Mantissas test stats
        '''
        print("\n", '  Mantissas Test  '.center(52, '#'))
        print(f"\nThe Mantissas MEAN is      {self.stats['Mean']:.6f}."
              "\tRef: 0.5")
        print(f"The Mantissas VARIANCE is  {self.stats['Var']:.6f}."
              "\tRef: 0.08333")
        print(f"The Mantissas SKEWNESS is  {self.stats['Skew']:.6f}."
              "\tRef: 0.0")
        print(f"The Mantissas KURTOSIS is  {self.stats['Kurt']:.6f}."
              "\tRef: -1.2\n")

    def show_plot(self, figsize=(12, 6)):
        '''
        plots the ordered mantissas and a line with the expected
                inclination. Defaults to True.

        Parameters
        ----------

        figsize -> tuple that sets the figure size
        '''
        ld = len(self.data)
        x = np.arange(1, ld + 1)
        n = np.ones(ld) / ld
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.plot(x, self.data.Mantissa.sort_values(), linestyle='--',
                color=colors['s'], linewidth=3, label='Mantissas')
        ax.plot(x, n.cumsum(), color=colors['m'],
                linewidth=2, label='Expected')
        plt.ylim((0, 1.))
        plt.xlim((1, ld + 1))
        ax.set_facecolor(colors['b'])
        ax.set_title("Ordered Mantissas")
        plt.legend(loc='upper left')
        plt.show();

    def arc_test(self, decimals=2, grid=True, figsize=12):
        '''
        Add two columns to Mantissas's DataFrame equal to their "X" and "Y"
        coordinates, plots its to a scatter plot and calculates the gravity
        center of the circle.

        Parameters
        ----------

        decimals -> number of decimal places for displaying the gravity center.
            Defaults to 2.
        
        grid -> show grid of the plot. Defaluts to True.
        
        figsize -> size of the figure to be displayed. Since it is a square,
            there is no need to provide a tuple, like is usually the case with
            matplotlib.
        '''
        if self.stats.get('gravity_center') is None:
            self.data['mant_x'] = np.cos(2 * np.pi * self.data.Mantissa)
            self.data['mant_y'] = np.sin(2 * np.pi * self.data.Mantissa)
            self.stats['gravity_center'] = (self.data.mant_x.mean(),
                                            self.data.mant_y.mean())
        fig = plt.figure(figsize=(figsize,figsize))
        ax = plt.subplot()
        ax.set_facecolor(colors['b'])
        ax.scatter(self.data.mant_x, self.data.mant_y, label= "ARC TEST",
                   color=colors['m'])
        ax.scatter(self.stats['gravity_center'][0], self.stats['gravity_center'][1],
                   color=colors['s']) 
        text_annotation = Annotation(
                    "  Gravity Center: "
                    f"x({round(self.stats['gravity_center'][0], decimals)}),"
                    f" y({round(self.stats['gravity_center'][1], decimals)})", 
                    xy=(self.stats['gravity_center'][0] - 0.65,
                        self.stats['gravity_center'][1] - 0.1),
                    xycoords='data')
        ax.add_artist(text_annotation)
        ax.grid(True, which='both')
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        ax.legend(loc = 'lower left')
        ax.set_title("Mantissas Arc Test")
        plt.show();


class Roll_mad(pd.Series):
    '''
    Applies the MAD to sequential subsets of the Series, returning another
    Series.

    Parameters
    ----------

    data: sequence of numbers to be evaluated. Must be a numpy 1D array,
        a pandas Series or a pandas DataFrame column, with values being
        integers or floats.

    test: tells which test to use. 1: Fisrt Digits; 2: First Two Digits;
        3: First Three Digits; 22: Second Digit; and -2: Last Two Digits.

    window: size of the subset to be used.

        decimals: number of decimal places to consider. Defaluts to 2.
        If integers, set to 0. If set to -infer-, it will remove the zeros
        and consider up to the fifth decimal place to the right, but will
        loose performance.


    sign: tells which portion of the data to consider. pos: only the positive
        entries; neg: only negative entries; all: all entries but zeros.
        Defaults to all.
    '''

    def __init__(self, data, test, window, decimals=2, sign='all'):

        test = _check_test_(test)

        if not isinstance(data, Source):
            start = Source(data, sign=sign, decimals=decimals, verbose=False)

        Exp, ind = _prep_to_roll_(start, test)

        pd.Series.__init__(self, start[digs_dict[test]].rolling(
            window=window).apply(_mad_to_roll_, args=(Exp, ind), raw=False))

        self.dropna(inplace=True)

        self.test = test

    def show_plot(self, test, figsize=(15, 8)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor(colors['b'])
        ax.plot(self, color=colors['m'])
        if test != -2:
            plt.axhline(y=mad_dict[test][0], color=colors['af'], linewidth=3)
            plt.axhline(y=mad_dict[test][1], color=colors['h2'], linewidth=3)
            plt.axhline(y=mad_dict[test][2], color=colors['s'], linewidth=3)
        plt.show()


class Roll_mse(pd.Series):
    '''
    Applies the MSE to sequential subsets of the Series, returning another
    Series.

    Parameters
    ----------

    data: sequence of numbers to be evaluated. Must be a numpy 1D array,
        a pandas Series or a pandas DataFrame column, with values being
        integers or floats.

    test: tells which test to use. 1: Fisrt Digits; 2: First Two Digits;
        3: First Three Digits; 22: Second Digit; and -2: Last Two Digits.

    window: size of the subset to be used.

        decimals: number of decimal places to consider. Defaluts to 2.
        If integers, set to 0. If set to -infer-, it will remove the zeros
        and consider up to the fifth decimal place to the right, but will
        loose performance.


    sign: tells which portion of the data to consider. pos: only the positive
        entries; neg: only negative entries; all: all entries but zeros.
        Defaults to all.
    '''

    def __init__(self, data, test, window, decimals=2, sign='all'):

        test = _check_test_(test)

        if not isinstance(data, Source):
            start = Source(data, sign=sign, decimals=decimals, verbose=False)

        Exp, ind = _prep_to_roll_(start, test)

        pd.Series.__init__(self, start[digs_dict[test]].rolling(
            window=window).apply(_mse_to_roll_, args=(Exp, ind), raw=False))

        self.dropna(inplace=True)

    def show_plot(self, figsize=(15, 8)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor(colors['b'])
        ax.plot(self, color=colors['m'])
        plt.show()


def _Z_score(frame, N):
    '''
    Returns the Z statistics for the proportions assessed

    frame -> DataFrame with the expected proportions and the already calculated
            Absolute Diferences between the found and expeccted proportions
    N -> sample size
    '''
    return (frame.AbsDif - (1 / (2 * N))) / np.sqrt(
           (frame.Expected * (1. - frame.Expected)) / N)


def _chi_square_(frame, ddf, confidence, verbose=True):
    '''
    Returns the chi-square statistic of the found distributions and compares
    it with the critical chi-square of such a sample, according to the
    confidence level chosen and the degrees of freedom - len(sample) -1.

    Parameters
    ----------
    frame:      DataFrame with Found, Expected and their difference columns.

    ddf:        Degrees of freedom to consider.

    confidence: Confidence level - confs dict.

    verbose:     prints the chi-squre result and compares to the critical
    chi-square for the sample. Defaults to True.
    '''
    if confidence is None:
        print('\nChi-square test needs confidence other than None.')
        return
    else:
        exp_counts = frame.Counts.sum() * frame.Expected
        dif_counts = frame.Counts - exp_counts
        found_chi = (dif_counts ** 2 / exp_counts).sum()
        crit_chi = crit_chi2[ddf][confidence]
        if verbose:
            print(f"\nThe Chi-square statistic is {found_chi}.\n"
                  f"Critical Chi-square for this series: {crit_chi}.")
        return (found_chi, crit_chi)


def _chi_square_2(frame):
    '''
    Returns the chi-square statistic of the found distributions

    Parameters
    ----------
    frame:      DataFrame with Found, Expected and their difference columns.

    '''
    exp_counts = frame.Counts.sum() * frame.Expected
    dif_counts = frame.Counts - exp_counts
    return (dif_counts ** 2 / exp_counts).sum()


def _KS_(frame, confidence, N, verbose=True):
    '''
    Returns the Kolmogorov-Smirnov test of the found distributions
    and compares it with the critical chi-square of such a sample,
    according to the confidence level chosen.

    Parameters
    ----------
    frame: DataFrame with Foud and Expected distributions.

    confidence: Confidence level - confs dict.

    N: Sample size

    verbose: prints the KS result and the critical value for the sample.
        Defaults to True.
    '''
    if confidence is None:
        print('\nKolmogorov-Smirnov test needs confidence other than None.')
        return
    else:
        # sorting and calculating the cumulative distribution
        ks_frame = frame.sort_index()[['Found', 'Expected']].cumsum()
        # finding the supremum - the largest cumul dist difference
        suprem = ((ks_frame.Found - ks_frame.Expected).abs()).max()
        # calculating the crittical value according to confidence
        crit_KS = KS_crit[confidence] / np.sqrt(N)

        if verbose:
            print(f"\nThe Kolmogorov-Smirnov statistic is {suprem}.\n"
                  f"Critical K-S for this series: {crit_KS}")
        return (suprem, crit_KS)


def _KS_2(frame):
    '''
    Returns the Kolmogorov-Smirnov test of the found distributions.

    Parameters
    ----------
    frame: DataFrame with Foud and Expected distributions.
    '''
    # sorting and calculating the cumulative distribution
    ks_frame = frame.sort_index()[['Found', 'Expected']].cumsum()
    # finding the supremum - the largest cumul dist difference
    return ((ks_frame.Found - ks_frame.Expected).abs()).max()


def _mad_(frame, test, verbose=True):
    '''
    Returns the Mean Absolute Deviation (MAD) between the found and the
    expected proportions.

    Parameters
    ----------

    frame: DataFrame with the Absolute Deviations already calculated.

    test: Test to compute the MAD from (F1D, SD, F2D...)

    verbose: prints the MAD result and compares to limit values of
        conformity. Defaults to True.
    '''
    mad = frame.AbsDif.mean()

    if verbose:
        print(f"\nThe Mean Absolute Deviation is {mad}")

        if test != -2:
            print(f"For the {mad_dict[digs_dict[test]]}:\n\
            - 0.0000 to {mad_dict[test][0]}: Close Conformity\n\
            - {mad_dict[test][0]} to {mad_dict[test][1]}: Acceptable Conformity\n\
            - {mad_dict[test][1]} to {3}: Marginally Acceptable Conformity\n\
            - Above {mad_dict[test][2]}: Nonconformity")
        else:
            pass
    return mad


def _mse_(frame, verbose=True):
    '''
    Returns the test's Mean Square Error

    frame -> DataFrame with the already computed Absolute Deviations between
            the found and expected proportions

    verbose -> Prints the MSE. Defaults to True. If False, returns MSE.
    '''
    mse = (frame.AbsDif ** 2).mean()

    if verbose:
        print(f"\nMean Square Error = {mse}")

    return mse


def _getMantissas_(arr):
    '''
    Returns the  mantissas, the non-integer part of the log of a number.

    arr: np.array of integers or floats ---> np.array of floats
    '''
    log_a = np.abs(np.log10(arr))
    return log_a - log_a.astype(int)  # the number - its integer part


def _lt_(num=False):
    '''
    Creates an array with the possible last two digits

    Parameters
    ----------

    num: returns numeric (ints) values. Defaluts to False,
        which returns strings.
    '''
    if num:
        n = np.arange(0, 100)
    else:
        n = np.arange(0, 100).astype(str)
        n[:10] = np.array(['00', '01', '02', '03', '04', '05',
                           '06', '07', '08', '09'])
    return n


def _plot_expected_(df, digs):
    '''
    Plots the Expected Benford Distributions

    df   -> DataFrame with the Expected Proportions
    digs -> Test's digit
    '''
    if digs in [1, 2, 3]:
        y_max = (df.Expected.max() + (10 ** -(digs) / 3)) * 100
        fig, ax = plt.subplots(figsize=(2 * (digs ** 2 + 5), 1.5 *
                                        (digs ** 2 + 5)))
    elif digs == 22:
        y_max = 13.
        fig, ax = plt.subplots(figsize=(14, 10.5))
    elif digs == -2:
        y_max = 1.1
        fig, ax = plt.subplots(figsize=(15, 8))
    plt.title('Expected Benford Distributions', size='xx-large')
    plt.xlabel(df.index.name, size='x-large')
    plt.ylabel('Distribution (%)', size='x-large')
    ax.set_facecolor(colors['b'])
    ax.set_ylim(0, y_max)
    ax.bar(df.index, df.Expected * 100, color=colors['t'], align='center')
    ax.set_xticks(df.index)
    ax.set_xticklabels(df.index)
    plt.show()

def _get_plot_args(digs):
    '''
    Gets the correct arguments for the plotting functions, depending on the
    the test (digs) chosen.
    '''
    if digs in [1, 2, 3]:
        text_x = False
        n, m = 10 ** (digs - 1), 10 ** (digs)
        x = np.arange(n, m)
        figsize = (2 * (digs ** 2 + 5), 1.5 * (digs ** 2 + 5))
    elif digs == 22:
        text_x = False
        x = np.arange(10)
        figsize = (14, 10)
    else:
        text_x = True
        x = np.arange(100)
        figsize = (15, 7)
    return x, figsize, text_x
    

def _plot_dig_(df, x, y_Exp, y_Found, N, figsize, conf_Z, text_x=False):
    '''
    Plots the digits tests results

    df -> DataFrame with the data to be plotted
    x -> sequence to be used in the x axis
    y_Exp -> sequence of the expected proportions to be used in the y axis
        (line)
    y_Found -> sequence of the found proportions to be used in the y axis
        (bars)
    N -> lenght of sequence, to be used when plotting the confidence levels
    figsize - > tuple to state the size of the plot figure
    conf_Z -> Confidence level
    text_x -> Forces to show all x ticks labels. Defaluts to True.
    '''
    if len(x) > 10:
        rotation = 90
    else:
        rotation = 0
    fig, ax = plt.subplots(figsize=figsize)
    plt.title('Expected vs. Found Distributions', size='xx-large')
    plt.xlabel('Digits', size='x-large')
    plt.ylabel('Distribution (%)', size='x-large')
    if conf_Z is not None:
        sig = conf_Z * np.sqrt(y_Exp * (1 - y_Exp) / N)
        upper = y_Exp + sig + (1 / (2 * N))
        lower_zeros = np.array([0]*len(upper))
        lower = np.maximum(y_Exp - sig - (1 / (2 * N)), lower_zeros)
        u = (y_Found < lower) | (y_Found > upper)
        c = np.array([colors['m']] * len(u))
        c[u] = colors['af']
        lower *= 100.
        upper *= 100.
        ax.plot(x, upper, color=colors['s'], zorder=5)
        ax.plot(x, lower, color=colors['s'], zorder=5)
        ax.fill_between(x, upper, lower, color=colors['s'],
                        alpha=.3, label='Conf')
    else:
        c = colors['m']
    ax.bar(x, y_Found * 100., color=c, label='Found', zorder=3, align='center')
    ax.plot(x, y_Exp * 100., color=colors['s'], linewidth=2.5,
            label='Benford', zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(x, rotation=rotation)
    ax.set_facecolor(colors['b'])
    if text_x:
        ind = np.array(df.index).astype(str)
        ind[:10] = np.array(['00', '01', '02', '03', '04', '05',
                             '06', '07', '08', '09'])
        plt.xticks(x, ind, rotation='vertical')
    ax.legend()
    ax.set_ylim(0, max([y_Exp.max() * 100, y_Found.max() * 100]) + 10 / len(x))
    ax.set_xlim(x[0] - 1, x[-1] + 1)
    plt.show()


def _plot_sum_(df, figsize, li, text_x=False):
    '''
    Plots the summation test results

    df -> DataFrame with the data to be plotted

    figsize - > tuple to state the size of the plot figure

    li -> values with which to draw the horizontal line
    '''
    x = df.index
    rotation = 90 if len(x) > 10 else 0
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    plt.title('Expected vs. Found Sums')
    plt.xlabel('Digits')
    plt.ylabel('Sums')
    ax.bar(x, df.Percent, color=colors['m'],
           label='Found Sums', zorder=3, align='center')
    ax.axhline(li, color=colors['s'], linewidth=2, label='Expected', zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(x, rotation=rotation)
    ax.set_facecolor(colors['b'])
    if text_x:
        ind = np.array(x).astype(str)
        ind[:10] = np.array(['00', '01', '02', '03', '04', '05',
                             '06', '07', '08', '09'])
        plt.xticks(x, ind, rotation='vertical')
    ax.legend()
    plt.show()


def _set_N_(len_df, limit_N):
    # Assigning to N the superior limit or the lenght of the series
    if limit_N is None or limit_N > len_df:
        return len_df
    # Check on limit_N being a positive integer
    else:
        if limit_N < 0 or not isinstance(limit_N, int):
            raise ValueError("limit_N must be None or a positive integer.")
        else:
            return limit_N


def _test_(digs):
    '''
    Returns the base instance for the proper test to be performed
    depending on the digit
    '''
    if digs in [1, 2, 3]:
        return First(digs, plot=False)
    elif digs == 22:
        return Second(plot=False)
    else:
        return LastTwo(num=True, plot=False)


def _input_data_(given):
    '''
    '''
    if (type(given) == pd.Series) | (type(given) == np.ndarray):
        data = None
        chosen = given
    elif type(given) == tuple:
        if (type(given[0]) != pd.DataFrame) | (type(given[1]) != str):
            raise TypeError('The data tuple must be composed of a pandas '
                            'DataFrame and the name (str) of the chosen '
                            'column, in that order')
        data = given[0]
        chosen = given[0][given[1]]
    else:
        raise TypeError("Wrong data input type. Check docstring.")
    return data, chosen


def _prep_(data, digs, limit_N, simple=False, confidence=None):
    '''
    Transforms the original number sequence into a DataFrame reduced
    by the ocurrences of the chosen digits, creating other computed
    columns
    '''
    N = _set_N_(len(data), limit_N=limit_N)

    # get the number of occurrences of the digits
    v = data.value_counts()
    # get their relative frequencies
    p = data.value_counts(normalize=True)
    # crate dataframe from them
    dd = pd.DataFrame({'Counts': v, 'Found': p}).sort_index()
    # join the dataframe with the one of expected Benford's frequencies
    dd = _test_(digs).join(dd).fillna(0)
    # create column with absolute differences
    dd['Dif'] = dd.Found - dd.Expected
    dd['AbsDif'] = np.absolute(dd.Dif)
    if simple:
        del dd['Dif']
        return dd
    else:
        if confidence is not None:
            dd['Z_score'] = _Z_score(dd, N)
        return N, dd


def first_digits(data, digs, decimals=2, sign='all', verbose=True,
                 confidence=None, high_Z='pos', limit_N=None,
                 MAD=False, MSE=False, chi_square=False, KS=False,
                 show_plot=True, inform=None):
    '''
    Performs the Benford First Digits test on the series of
    numbers provided.

    Parameters
    ----------

    data: sequence of numbers to be evaluated. Must be a numpy 1D array,
        a pandas Series or a pandas DataFrame column, with values being
        integers or floats.

    decimals: number of decimal places to consider. Defaluts to 2.
        If integers, set to 0. If set to -infer-, it will remove the zeros
        and consider up to the fifth decimal place to the right, but will
        loose performance.

    sign: tells which portion of the data to consider. pos: only the positive
        entries; neg: only negative entries; all: all entries but zeros.
        Defaults to all.`

    digs: number of first digits to consider. Must be 1 (first digit),
        2 (first two digits) or 3 (first three digits).

    verbose: tells the number of registries that are being subjected to
        the analysis and returns tha analysis DataFrame sorted by the
        highest Z score down. Defaults to True.

    MAD: calculates the Mean Absolute Difference between the
        found and the expected distributions; defaults to False.

    confidence: confidence level to draw lower and upper limits when
        plotting and to limit the top deviations to show. Defaults to None.

    high_Z: chooses which Z scores to be used when displaying results,
        according to the confidence level chosen. Defaluts to 'pos',
        which will highlight only values higher than the expexted
        frequencies; 'all' will highlight both extremes (positive and
        negative); and an integer, which will use the first n entries,
        positive and negative, regardless of whether Z is higher than
        the confidence or not.

    limit_N: sets a limit to N as the sample size for the calculation of
        the Z scores if the sample is too big. Defaults to None.

    MSE: calculates the Mean Square Error of the sample; defaults to
        False.

    chi_square: calculates the chi_square statistic of the sample and
        compares it with a critical value, according to the confidence
        level chosen and the series's degrees of freedom. Defaults to
        False. Requires confidence != None.

    KS: calculates the Kolmogorov-Smirnov test, comparing the cumulative
        distribution of the sample with the Benford's, according to the
        confidence level chosen. Defaults to False. Requires confidence
        != None.

    show_plot: draws the test plot.
    '''
    verbose = _deprecate_inform_(verbose, inform)

    if not isinstance(data, Source):
        data = Source(data, decimals=decimals, sign=sign, verbose=verbose)

    data = data.first_digits(digs, verbose=verbose, confidence=confidence,
                             high_Z=high_Z, limit_N=limit_N, MAD=MAD, MSE=MSE,
                             chi_square=chi_square, KS=KS, show_plot=show_plot,
                             ret_df=True)

    if confidence is not None:
        data = data[['Counts', 'Found', 'Expected', 'Z_score']]
        return data.sort_values('Z_score', ascending=False)
    else:
        return data[['Counts', 'Found', 'Expected']]


def second_digit(data, decimals=2, sign='all', verbose=True,
                 confidence=None, high_Z='pos', limit_N=None,
                 MAD=False, MSE=False, chi_square=False, KS=False,
                 show_plot=True, inform=None):
    '''
    Performs the Benford Second Digits test on the series of
    numbers provided.

    Parameters
    ----------

    data: sequence of numbers to be evaluated. Must be a numpy 1D array,
        a pandas Series or a pandas DataFrame column, with values being
        integers or floats.

    decimals: number of decimal places to consider. Defaluts to 2.
        If integers, set to 0. If set to -infer-, it will remove the zeros
        and consider up to the fifth decimal place to the right, but will
        loose performance.

    sign: tells which portion of the data to consider. pos: only the positive
        entries; neg: only negative entries; all: all entries but zeros.
        Defaults to all.`

    verbose: tells the number of registries that are being subjected to
        the analysis and returns tha analysis DataFrame sorted by the
        highest Z score down. Defaults to True.

    MAD: calculates the Mean Absolute Difference between the
        found and the expected distributions; defaults to False.

    confidence: confidence level to draw lower and upper limits when
        plotting and to limit the top deviations to show. Defaults to None.

    high_Z: chooses which Z scores to be used when displaying results,
        according to the confidence level chosen. Defaluts to 'pos',
        which will highlight only values higher than the expexted
        frequencies; 'all' will highlight both extremes (positive and
        negative); and an integer, which will use the first n entries,
        positive and negative, regardless of whether Z is higher than
        the confidence or not.

    limit_N: sets a limit to N as the sample size for the calculation of
        the Z scores if the sample is too big. Defaults to None.

    MSE: calculates the Mean Square Error of the sample; defaults to
        False.

    chi_square: calculates the chi_square statistic of the sample and
        compares it with a critical value, according to the confidence
        level chosen and the series's degrees of freedom. Defaults to
        False. Requires confidence != None.

    KS: calculates the Kolmogorov-Smirnov test, comparing the cumulative
        distribution of the sample with the Benford's, according to the
        confidence level chosen. Defaults to False. Requires confidence
        != None.

    show_plot: draws the test plot.

    '''
    verbose = _deprecate_inform_(verbose, inform)

    if not isinstance(data, Source):
        data = Source(data, sign=sign, decimals=decimals, verbose=verbose)

    data = data.second_digit(verbose=verbose, confidence=confidence,
                             high_Z=high_Z, limit_N=limit_N, MAD=MAD, MSE=MSE,
                             chi_square=chi_square, KS=KS, show_plot=show_plot,
                             ret_df=True)
    if confidence is not None:
        data = data[['Counts', 'Found', 'Expected', 'Z_score']]
        return data.sort_values('Z_score', ascending=False)
    else:
        return data[['Counts', 'Found', 'Expected']]


def last_two_digits(data, decimals=2, sign='all', verbose=True,
                    confidence=None, high_Z='pos', limit_N=None,
                    MAD=False, MSE=False, chi_square=False, KS=False,
                    show_plot=True, inform=None):
    '''
    Performs the Last Two Digits test on the series of
    numbers provided.

    Parameters
    ----------

    data: sequence of numbers to be evaluated. Must be a numpy 1D array,
        a pandas Series or a pandas DataFrame column,with values being
        integers or floats.

    decimals: number of decimal places to consider. Defaluts to 2.
        If integers, set to 0. If set to -infer-, it will remove the zeros
        and consider up to the fifth decimal place to the right, but will
        loose performance.

    sign: tells which portion of the data to consider. pos: only the positive
        entries; neg: only negative entries; all: all entries but zeros.
        Defaults to all.`

    verbose: tells the number of registries that are being subjected to
        the analysis and returns tha analysis DataFrame sorted by the
        highest Z score down. Defaults to True.

    confidence: confidence level to draw lower and upper limits when
        plotting and to limit the top deviations to show. Defaults to None.

    high_Z: chooses which Z scores to be used when displaying results,
        according to the confidence level chosen. Defaluts to 'pos',
        which will highlight only values higher than the expexted
        frequencies; 'all' will highlight both extremes (positive and
        negative); and an integer, which will use the first n entries,
        positive and negative, regardless of whether Z is higher than
        the confidence or not.

    limit_N: sets a limit to N as the sample size for the calculation of
        the Z scores if the sample is too big. Defaults to None.

    MAD: calculates the Mean Absolute Difference between the
        found and the expected distributions; defaults to False.

    MSE: calculates the Mean Square Error of the sample; defaults to
        False.

    chi_square: calculates the chi_square statistic of the sample and
        compares it with a critical value, according to the confidence
        level chosen and the series's degrees of freedom. Defaults to
        False. Requires confidence != None.

    KS: calculates the Kolmogorov-Smirnov test, comparing the cumulative
        distribution of the sample with the Benford's, according to the
        confidence level chosen. Defaults to False. Requires confidence
        != None.

    show_plot: draws the test plot.

    '''
    verbose = _deprecate_inform_(verbose, inform)

    if not isinstance(data, Source):
        data = Source(data, decimals=decimals, sign=sign, verbose=verbose)

    data = data.last_two_digits(verbose=verbose, confidence=confidence,
                                high_Z=high_Z, limit_N=limit_N, MAD=MAD,
                                MSE=MSE, chi_square=chi_square, KS=KS,
                                show_plot=show_plot, ret_df=True)

    if confidence is not None:
        data = data[['Counts', 'Found', 'Expected', 'Z_score']]
        return data.sort_values('Z_score', ascending=False)
    else:
        return data[['Counts', 'Found', 'Expected']]


def mantissas(data, report=True, show_plot=True, arc_test=True, inform=None):
    '''
    Returns a Series with the data mantissas,

    Parameters
    ----------
    data: sequence to compute mantissas from, numpy 1D array, pandas
        Series of pandas DataFrame column.

    report: prints the mamtissas mean, variance, skewness and kurtosis
        for the sequence studied, along with reference values.

    show_plot: plots the ordered mantissas and a line with the expected
        inclination. Defaults to True.
    '''
    report = _deprecate_inform_(report, inform)

    mant = Mantissas(data)
    if report:
        mant.report()
    if show_plot:
        mant.show_plot()
    if arc_test:
        mant.arc_test()
    return mant


def summation(data, digs=2, decimals=2, sign='all', top=20, verbose=True,
              show_plot=True, inform=None):
    '''
    Performs the Summation test. In a Benford series, the sums of the
    entries begining with the same digits tends to be the same.
    Works only with the First Digits (1, 2 or 3) test.

    Parameters
    ----------

    digs: tells the first digits to use: 1- first; 2- first two;
        3- first three. Defaults to 2.

    decimals: number of decimal places to consider. Defaluts to 2.
        If integers, set to 0. If set to -infer-, it will remove the zeros
        and consider up to the fifth decimal place to the right, but will
        loose performance.

    top: choses how many top values to show. Defaults to 20.

    show_plot: plots the results. Defaults to True.

    '''
    verbose = _deprecate_inform_(verbose, inform)

    if not isinstance(data, Source):
        data = Source(data, sign=sign, decimals=decimals, verbose=verbose)

    data = data.summation(digs=digs, top=top, verbose=verbose,
                          show_plot=show_plot, ret_df=True)
    if verbose:
        return data.sort_values('AbsDif', ascending=False)
    else:
        return data


def mad(data, test, decimals=2, sign='all'):
    '''
    Returns the Mean Absolute Deviation of the Series

    Parameters
    ----------

    data: sequence of numbers to be evaluated. Must be a numpy 1D array,
        a pandas Series or a pandas DataFrame column, with values being
        integers or floats.

    test: informs which base test to use for the mad.

    decimals: number of decimal places to consider. Defaluts to 2.
        If integers, set to 0. If set to -infer-, it will remove the zeros
        and consider up to the fifth decimal place to the right, but will
        loose performance.

    sign: tells which portion of the data to consider. pos: only the positive
        entries; neg: only negative entries; all: all entries but zeros.
        Defaults to all.`

    '''
    _check_test_(test)
    start = Source(data.values, sign=sign, decimals=decimals, verbose=False)
    if test in [1, 2, 3]:
        start.first_digits(digs=test, MAD=True, MSE=True, simple=True)
    elif test == 22:
        start.second_digit(MAD=True, MSE=False, simple=True)
    else:
        start.last_two_digits(MAD=True, MSE=False, simple=True)
    return start.MAD


def mse(data, test, decimals=2, sign='all'):
    '''
    Returns the Mean Squared Error of the Series
    '''
    test = _check_test_(test)
    start = Source(data, sign=sign, decimals=decimals, verbose=False)
    if test in [1, 2, 3]:
        start.first_digits(digs=test, MAD=False, MSE=True, simple=True)
    elif test == 22:
        start.second_digit(MAD=False, MSE=True, simple=True)
    else:
        start.last_two_digits(MAD=False, MSE=True, simple=True)
    return start.MSE


def mad_summ(data, test, decimals=2, sign='all'):
    '''
    Returns the Mean Absolute Deviation of the Summation Test

    Parameters
    ----------

    data: sequence of numbers to be evaluated. Must be a numpy 1D array,
        a pandas Series or a pandas DataFrame column, with values being
        integers or floats.

    test: informs which base test to use for the summation mad.

    decimals: number of decimal places to consider. Defaluts to 2.
        If integers, set to 0. If set to -infer-, it will remove the zeros
        and consider up to the fifth decimal place to the right, but will
        loose performance.

    sign: tells which portion of the data to consider. pos: only the positive
        entries; neg: only negative entries; all: all entries but zeros.
        Defaults to all.`

    '''
    _check_digs_(test)

    start = Source(data, sign=sign, decimals=decimals, verbose=False)
    temp = start.loc[start.ZN >= 10 ** (test - 1)]
    temp[digs_dict[test]] = (temp.ZN // 10 ** ((np.log10(temp.ZN).astype(
                                                int)) - (test - 1))).astype(
                                                    int)
    li = 1. / (9 * (10 ** (test - 1)))

    df = temp.groupby(digs_dict[test]).sum()
    return np.mean(np.absolute(df.ZN / df.ZN.sum() - li))


def _prep_to_roll_(start, test):
    '''
    Used by the rolling mad and rolling mean, prepares each test and
    respective expected proportions for later application to the Series subset
    '''
    if test in [1, 2, 3]:
        start[digs_dict[test]] = start.ZN // 10 ** ((
            np.log10(start.ZN).astype(int)) - (test - 1))
        start = start.loc[start.ZN >= 10 ** (test - 1)]

        ind = np.arange(10 ** (test - 1), 10 ** test)
        Exp = np.log10(1 + (1. / ind))

    elif test == 22:
        start[digs_dict[test]] = (start.ZN // 10 ** ((
            np.log10(start.ZN)).astype(int) - 1)) % 10
        start = start.loc[start.ZN >= 10]

        Expec = np.log10(1 + (1. / np.arange(10, 100)))
        temp = pd.DataFrame({'Expected': Expec, 'Sec_Dig':
                             np.array(list(range(10)) * 9)})
        Exp = temp.groupby('Sec_Dig').sum().values.reshape(10,)
        ind = np.arange(0, 10)

    else:
        start[digs_dict[test]] = start.ZN % 100
        start = start.loc[start.ZN >= 1000]

        ind = np.arange(0, 100)
        Exp = np.array([1 / 99.] * 100)

    return Exp, ind


def rolling_mad(data, test, window, decimals=2, sign='all', show_plot=False):
    '''
    Applies the MAD to sequential subsets of the Series, returning another
    Series.

    Parameters
    ----------

    data: sequence of numbers to be evaluated. Must be a numpy 1D array,
        a pandas Series or a pandas DataFrame column, with values being
        integers or floats.

    test: tells which test to use. 1: Fisrt Digits; 2: First Two Digits;
        3: First Three Digits; 22: Second Digit; and -2: Last Two Digits.

    window: size of the subset to be used.

    decimals: number of decimal places to consider. Defaluts to 2.
        If integers, set to 0. If set to -infer-, it will remove the zeros
        and consider up to the fifth decimal place to the right, but will
        loose performance.

    sign: tells which portion of the data to consider. pos: only the positive
        entries; neg: only negative entries; all: all entries but zeros.
        Defaults to all.

    show_plot: draws the test plot.
    '''
    test = _check_test_(test)
    r_mad = Roll_mad(data, test, window, decimals, sign)
    if show_plot:
        r_mad.show_plot(test)
    return r_mad


def _mad_to_roll_(arr, Exp, ind):
    '''
    Mean Absolute Deviation used in the rolling function
    '''
    prop = pd.Series(arr)
    prop = prop.value_counts(normalize=True).sort_index()

    if len(prop) < len(Exp):
        prop = prop.reindex(ind).fillna(0)

    return np.absolute(prop - Exp).mean()


def rolling_mse(data, test, window, decimals=2, sign='all', show_plot=False):
    '''
    Applies the MSE to sequential subsets of the Series, returning another
    Series.

    Parameters
    ----------

    data: sequence of numbers to be evaluated. Must be a numpy 1D array,
        a pandas Series or a pandas DataFrame column, with values being
        integers or floats.

    test: tells which test to use. 1: Fisrt Digits; 2: First Two Digits;
        3: First Three Digits; 22: Second Digit; and -2: Last Two Digits.

    window: size of the subset to be used.

    decimals: number of decimal places to consider. Defaluts to 2.
        If integers, set to 0. If set to -infer-, it will remove the zeros
        and consider up to the fifth decimal place to the right, but will
        loose performance.

    sign: tells which portion of the data to consider. pos: only the positive
        entries; neg: only negative entries; all: all entries but zeros.
        Defaults to all.

    show_plot: draws the test plot.
    '''
    r_mse = Roll_mse(data, test, window, decimals, sign)
    if show_plot:
        r_mse.show_plot()
    return r_mse


def _mse_to_roll_(arr, Exp, ind):
    '''
    Mean Squared Error used in the rolling function
    '''
    prop = pd.Series(arr)
    temp = prop.value_counts(normalize=True).sort_index()

    if len(temp) < len(Exp):
        temp = temp.reindex(ind).fillna(0)

    return ((temp - Exp) ** 2).mean()


def duplicates(data, top_Rep=20, verbose=True, inform=None):
    '''
    Performs a duplicates test and maps the duplicates count in descending
    order.

    Parameters
    ----------
    data: sequence to take the duplicates from. pandas Series or
        numpy Ndarray.

    verbose: tells how many duplicated entries were found and prints the
        top numbers according to the top_Rep parameter. Defaluts to True.

    top_Rep: chooses how many duplicated entries will be
        shown withe the top repititions. int or None. Defaluts to 20.
        If None, returns al the ordered repetitions.
    '''
    verbose = _deprecate_inform_(verbose, inform)

    if top_Rep is not None and not isinstance(top_Rep, int):
        raise ValueError('The top_Rep parameter must be an int or None.')

    if not isinstance(data, pd.Series):
        try:
            data = pd.Series(data)
        except ValueError:
            print('\ndata must be a numpy Ndarray or a pandas Series.')

    dup = data.loc[data.duplicated(keep=False)]
    dup_count = dup.value_counts()

    dup_count.index.names = ['Entries']
    dup_count.name = 'Count'

    if verbose:
        print(f'\nFound {len(dup_count)} duplicated entries.\n'
              f'The entries with the {top_Rep} highest repitition counts are:')
        print(dup_count.head(top_Rep))

    return dup_count


def second_order(data, test, decimals=2, sign='all', verbose=True, MAD=False,
                 confidence=None, high_Z='pos', limit_N=None, MSE=False,
                 show_plot=True, inform=None):
    '''
    Performs the chosen test after subtracting the ordered sequence by itself.
    Hence Second Order.

    Parameters
    ----------

    data: sequence of numbers to be evaluated. Must be a numpy 1D array,
        a pandas Series or a pandas DataFrame column, with values being
        integers or floats.

    test: the test to be performed - 1 or 'F1D': First Digit; 2 or 'F2D':
        First Two Digits; 3 or 'F3D': First three Digits; 22 or 'SD':
        Second Digits; -2 or 'L2D': Last Two Digits.

    decimals: number of decimal places to consider. Defaluts to 2.
        If integers, set to 0. If set to -infer-, it will remove the zeros
        and consider up to the fifth decimal place to the right, but will
        loose performance.

    sign: tells which portion of the data to consider. pos: only the positive
        entries; neg: only negative entries; all: all entries but zeros.
        Defaults to all.`

    verbose: tells the number of registries that are being subjected to
        the analysis and returns tha analysis DataFrame sorted by the
        highest Z score down. Defaults to True.

    MAD: calculates the Mean Absolute Difference between the
        found and the expected distributions; defaults to False.

    confidence: confidence level to draw lower and upper limits when
        plotting and to limit the top deviations to show. Defaults to None.

    high_Z: chooses which Z scores to be used when displaying results,
        according to the confidence level chosen. Defaluts to 'pos',
        which will highlight only values higher than the expexted
        frequencies; 'all' will highlight both extremes (positive and
        negative); and an integer, which will use the first n entries,
        positive and negative, regardless of whether Z is higher than
        the confidence or not.

    limit_N: sets a limit to N as the sample size for the calculation of
        the Z scores if the sample is too big. Defaults to None.

    MSE: calculates the Mean Square Error of the sample; defaults to
        False.

    chi_square: calculates the chi_square statistic of the sample and
        compares it with a critical value, according to the confidence
        level chosen and the series's degrees of freedom. Defaults to
        False. Requires confidence != None.

    KS: calculates the Kolmogorov-Smirnov test, comparing the cumulative
        distribution of the sample with the Benford's, according to the
        confidence level chosen. Defaults to False. Requires confidence
        != None.

    show_plot: draws the test plot.
    '''
    test = _check_test_(test)

    verbose = _deprecate_inform_(verbose, inform)

    data = Source(data, decimals=decimals, sign=sign,
                  sec_order=True, verbose=verbose)
    if test in [1, 2, 3]:
        data.first_digits(digs=test, verbose=verbose, MAD=MAD,
                          confidence=confidence, high_Z=high_Z,
                          limit_N=limit_N, MSE=MSE, show_plot=show_plot)
    elif test == 22:
        data.second_digit(verbose=verbose, MAD=MAD, confidence=confidence,
                          high_Z=high_Z, limit_N=limit_N, MSE=MSE,
                          show_plot=show_plot)
    else:
        data.last_two_digits(verbose=verbose, MAD=MAD,
                             confidence=confidence, high_Z=high_Z,
                             limit_N=limit_N, MSE=MSE, show_plot=show_plot)
    return data


def _check_digs_(digs):
    '''
    Chhecks the possible values for the digs of the First Digits test1
    '''
    if digs not in [1, 2, 3]:
        raise ValueError("The value assigned to the parameter -digs- "
                         f"was {digs}. Value must be 1, 2 or 3.")


def _check_test_(test):
    '''
    Checks the test chosen, both for int or str values
    '''
    if isinstance(test, int):
        if test in digs_dict.keys():
            return test
        else:
            raise ValueError(f'Test was set to {test}. Should be one of '
                             f'{digs_dict.keys()}')
    elif isinstance(test, str):
        if test in rev_digs.keys():
            return rev_digs[test]
        else:
            raise ValueError(f'Test was set to {test}. Should be one of '
                             f'{rev_digs.keys()}')
    else:
        raise ValueError('Wrong value chosen for test parameter. Possible '
                         f'values are\n {list(digs_dict.keys())} for ints and'
                         f'\n {list(rev_digs.keys())} for strings.')

def _check_confidence_(confidence):
    '''
    '''
    if confidence not in confs.keys():
        raise ValueError("Value of parameter -confidence- must be one of the "
                         f"following:\n {list(confs.keys())}")
    return confidence

def _check_high_Z_(high_Z):
    '''
    '''
    if not high_Z in ['pos', 'all']:
        if not isinstance(high_Z, int):
            raise ValueError("The parameter -high_Z- should be 'pos', "
                             "'all' or an int.")
    return high_Z

def _check_num_array(data):
    '''
    '''
    if (not isinstance(data, np.ndarray)) & (not isinstance(data, pd.Series)):
        print('\n`data` not a numpy NDarray nor a pandas Series.'
                ' Trying to convert...')
        try:
            data = np.array(data)
        except:
            raise ValueError('Could not convert data. Check input.')
        print('\nConversion successful.')
    elif (data.dtype == int) | (not data.dtype == float):
        print("\n`data` type not int nor float. Trying to convert...")
        try:
            data = data.astype(float)
        except:
            raise ValueError('Could not convert data. Check input.')
    return data


def _subtract_sorted_(data):
    '''
    Subtracts the sorted sequence elements from each other, discarding zeros.
    Used in the Second Order test
    '''
    sec = data.copy()
    sec.sort_values(inplace=True)
    sec = sec - sec.shift(1)
    sec = sec.loc[sec != 0]
    return sec


def _inform_(df, high_Z, conf):
    '''
    Selects and sorts by the Z_stats chosen to be considered, informing or not.
    '''

    if isinstance(high_Z, int):
        if conf is not None:
            dd = df[['Expected', 'Found', 'Z_score'
                     ]].sort_values('Z_score', ascending=False).head(high_Z)
            print(f'\nThe entries with the top {high_Z} Z scores are:\n')
        # Summation Test
        else:
            dd = df[['Expected', 'Found', 'AbsDif'
                     ]].sort_values('AbsDif', ascending=False
                                    ).head(high_Z)
            print(f'\nThe entries with the top {high_Z} absolute deviations '
                  'are:\n')
    else:
        if high_Z == 'pos':
            m1 = df.Dif > 0
            m2 = df.Z_score > conf
            dd = df[['Expected', 'Found', 'Z_score'
                     ]].loc[m1 & m2].sort_values('Z_score', ascending=False)
            print('\nThe entries with the significant positive '
                  'deviations are:\n')
        elif high_Z == 'neg':
            m1 = df.Dif < 0
            m2 = df.Z_score > conf
            dd = df[['Expected', 'Found', 'Z_score'
                     ]].loc[m1 & m2].sort_values('Z_score', ascending=False)
            print('\nThe entries with the significant negative '
                  'deviations are:\n')
        else:
            dd = df[['Expected', 'Found', 'Z_score'
                     ]].loc[df.Z_score > conf].sort_values('Z_score',
                                                           ascending=False)
            print('\nThe entries with the significant deviations are:\n')
    print(dd)

def _report_MAD_(digs, MAD):
    '''
    Reports the test Mean Absolut Deviation and compares it to critical values
    '''
    print(f'Mean Absolute Deviation: {MAD:.6f}')
    if digs != -2:
        mads = mad_dict[digs]
        if MAD <= mads[0]:
            print(f'MAD <= {mads[0]:.6f}: Close conformity.\n')
        elif MAD <= mads[1]:
            print(f'{mads[0]:.6f} < MAD <= {mads[1]:.6f}: '
                  'Acceptable conformity.\n')
        elif MAD <= mads[2]:
            print(f'{mads[1]:.6f} < MAD <= {mads[2]:.6f}: '
                  'Marginally Acceptable conformity.\n')
        else:
            print(f'MAD > {mads[2]:.6f}: Nonconformity.\n')
    else:
        print("There is no conformity check for this test's MAD.\n")

def _report_KS_(KS, crit_KS):
    '''
    Reports the test Kolmogorov Smirnoff statistic and compares it to critical
    values, depending on the confidence level
    '''
    result = 'PASS' if KS <= crit_KS else 'FAIL'
    print(f"\n\tKolmogorov Smirnoff: {KS:.6f}",
          f"\n\tCritical value: {crit_KS:.6f} -- {result}")

def _report_chi2_(chi2, crit_chi2):
    '''
    Reports the test Chi-square statistic and compares it to critical values,
    depending on the confidence level
    '''
    result = 'PASS' if chi2 <= crit_chi2 else 'FAIL'
    print(f"\n\tChi square: {chi2:.6f}",
          f"\n\tCritical value: {crit_chi2:.6f} -- {result}")

def _report_Z_(df, high_Z, crit_Z):
    '''
    Reports the test Z scores and compares them to a critical value,
    depending on the confidence level
    '''
    print(f"\n\tCritical Z-score:{crit_Z}.")
    _inform_(df, high_Z, crit_Z)

def _report_summ_(test, high_diff):
    '''
    Reports the Summation Test Absolute Differences between the Found and
    the Expected proportions

    '''
    if high_diff is not None:
        print(f'\nThe top {high_diff} Absolute Differences are:\n')
        print(test.sort_values('AbsDif', ascending=False).head(high_diff))
    else:
        print('\nThe top Absolute Differences are:\n')
        print(test.sort_values('AbsDif', ascending=False))
    

def _report_test_(test, high=None, crit_vals=None):
    '''
    Main report function. Receives the parameters to report with, initiates
    the process, and calls the right reporting helper function(s), depending
    on the Test.
    '''
    print(f'\n  {test.name}  '.center(50, '#'), '\n')
    if not 'Summation' in test.name:
        _report_MAD_(test.digs, test.MAD)
        if test.confidence is not None:
            print(f"For confidence level {test.confidence}%: ")
            _report_KS_(test.KS, crit_vals['KS'])
            _report_chi2_(test.chi_square, crit_vals['chi2'])
            _report_Z_(test, high, crit_vals['Z'])
        else:
            print('Confidence is currently `None`. Set the confidence level, '
                    'so as to generate comparable critical values.' )
            if isinstance(high, int):
                _inform_(test, high, None)
    else:
        _report_summ_(test, high)


def _deprecate_inform_(verbose, inform):
    if inform is None:
        return verbose
    else:
        warnings.warn('The parameter `inform` will be deprecated in future '
                      'versions. Use `verbose` instead.',
                      PendingDeprecationWarning)
        return inform