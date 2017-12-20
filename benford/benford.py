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


digs_dict = {1: 'F1D', 2: 'F2D', 3: 'F3D', 22: 'SD', -2: 'L2D'}

rev_digs = {'F1D': 1, 'F2D': 2, 'F3D': 3, 'SD': 22, 'L2D': -2}

mad_dict = {1: [0.006, 0.012, 0.015], 2: [0.0012, 0.0018, 0.0022],
            3: [0.00036, 0.00044, 0.00050], 22: [0.008, 0.01, 0.012],
            'F1D': 'First Digit', 'F2D': 'First Two Digits',
            'F3D': 'First Three Digits', 'SD': 'Second Digits'}

colors = {'m': '#00798c', 'b': '#E2DCD8', 's': '#9c3848',
          'af': '#edae49', 'ab': '#33658a', 'h': '#d1495b',
          'h2': '#f64740', 't': '#16DB93'}

confs = {'None': None, '80': 1.285, '85': 1.435, '90': 1.645, '95': 1.96,
         '99': 2.576, '99.9': 3.29, '99.99': 3.89, '99.999': 4.417,
         '99.9999': 4.892, '99.99999': 5.327}

crit_chi2 = {8: {'80': 11.03, '85': 12.027, '90': 13.362, '95': 15.507,
                 '99': 20.090, '99.9': 26.124, '99.99': 31.827,
                 '99.999': 37.332, '99.9999': 42.701, '99.99999': 47.972},
             9: {'80': 12.242, '85': 13.288, '90': 14.684, '95': 16.919,
                 '99': 21.666, '99.9': 27.877, '99.99': 33.72,
                 '99.999': 39.341, '99.9999': 44.811, '99.99999': 50.172},
             89: {'80': 99.991, '85': 102.826, '90': 106.469, '95': 112.022,
                  '99': 122.942, '99.9': 135.978, '99.99': 147.350,
                  '99.999': 157.702, '99.9999': 167.348, '99.99999': 176.471},
             99: {'80': 110.607, '85': 113.585, '90': 117.407,
                  '95': 123.225, '99': 134.642, '99.9': 148.230,
                  '99.99': 160.056, '99.999': 170.798, '99.9999': 180.792,
                  '99.99999': 190.23},
             899: {'80': 934.479, '85': 942.981, '90': 953.752, '95': 969.865,
                   '99': 1000.575, '99.9': 1035.753, '99.99': 1065.314,
                   '99.999': 1091.422, '99.9999': 1115.141,
                   '99.99999': 1137.082}
             }
KS_crit = {'80': 1.075, '85': 1.139, '90': 1.125, '95': 1.36, '99': 1.63,
           '99.9': 1.95, '99.99': 2.23, '99.999': 2.47,
           '99.9999': 2.7, '99.99999': 2.9}


class First(pd.DataFrame):
    '''
     Returns the expected probabilities of the First, First Two, or
     First Three digits according to Benford's distribution.

    Parameters
    ----------

    -> digs: 1, 2 or 3 - tells which of the first digits to consider:
            1 for the First Digit, 2 for the First Two Digits and 3 for
            the First Three Digits.

    -> plot: option to plot a bar chart of the Expected proportions.
            Defaults to True.
    '''

    def __init__(self, digs, plot=True):
        if digs not in [1, 2, 3]:
            raise ValueError("The value assigned to the parameter -digs-\
 was {0}. Value must be 1, 2 or 3.".format(digs))
        dig_name = 'First_{0}_Dig'.format(digs)
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


class Analysis(pd.DataFrame):
    '''
    Prepares the data for Analysis. pandas DataFrame subclass.

    Parameters
    ----------

    data: sequence of numbers to be evaluated. Must be a numpy 1D array,
        a pandas Series or a pandas DataFrame column, with values being
        integers or floats.

    decimals: number of decimal places to consider. Defaluts to 2.
        If integers, set to 0. If set to -infer-, it will deal separately
        (and differently) with each registry.

    sign: tells which portion of the data to consider. pos: only the positive
        entries; neg: only negative entries; all: all entries but zeros.
        Defaults to all.`

    sec_order: choice for the Second Order Test, which cumputes the
        differences between the ordered entries before running the Tests.

    inform: tells the number of registries that are being subjected to
        the Analysis; defaults to True
    '''

    def __init__(self, data, decimals=2, sign='all', sec_order=False,
                 inform=True):

        if sign not in ['all', 'pos', 'neg']:
            raise ValueError("The -sign- argument must be 'all','pos'\
 or 'neg'.")

        pd.DataFrame.__init__(self, {'Seq': data})

        if self.Seq.dtypes != 'float' and self.Seq.dtypes != 'int':
            raise TypeError("The sequence dtype was not int nor float.\n\
Convert it to whether int of float, and try again.")

        if sign == 'pos':
            self.Seq = self.Seq.loc[self.Seq > 0]
        elif sign == 'neg':
            self.Seq = self.Seq.loc[self.Seq < 0]
        else:
            self.Seq = self.Seq.loc[self.Seq != 0]

        self.dropna(inplace=True)

        if inform:
            print("Initialized sequence with {0} registries.".format(
                  len(self)))
        if sec_order:
            self.Seq = _subtract_sorted_(self.Seq.copy())
            self.dropna(inplace=True)
            self.reset_index(inplace=True)
            if inform:
                print('Second Order Test. Initial series reduced to {0}\
 entries.'.format(len(self.Seq)))

        ab = np.abs(self.Seq)

        if decimals == 'infer':
            # Getting the different number of decimal places
            decimals = (ab - ab.astype(int)
                        ).astype(str).str.strip('0.').str.len()

        self['ZN'] = (ab * (10**decimals)).astype(int)

    def mantissas(self, plot=True, figsize=(15, 8)):
        '''
        Calculates the mantissas, their mean and variance, and compares them
        with the mean and variance of a Benford's sequence.

        Parameters
        ----------

        plot: plots the ordered mantissas and a line with the expected
            inclination. Defaults to True.

        figsize -> tuple that sets the figure size
        '''
        self['Mant'] = _getMantissas_(self.Seq)
        p = self[['Seq', 'Mant']]
        p = p.loc[p.Seq > 0].sort_values('Mant')
        print("The Mantissas MEAN is {0}. Ref: 0.5.".format(p.Mant.mean()))
        print("The Mantissas VARIANCE is {0}. Ref: 0.083333.".format(
              p.Mant.var()))
        print("The Mantissas SKEWNESS is {0}. \tRef: 0.".format(p.Mant.skew()))
        print("The Mantissas KURTOSIS is {0}. \tRef: -1.2.".
              format(p.Mant.kurt()))

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

    def first_digits(self, digs, inform=True, confidence=None, high_Z='pos',
                     limit_N=None, MAD=False, MSE=False, chi_square=False, KS=False,
                     show_plot=True, simple=False, ret_df=False):
        '''
        Performs the Benford First Digits test with the series of
        numbers provided, and populates the mapping dict for future
        selection of the original series.

        digs -> number of first digits to consider. Must be 1 (first digit),
            2 (first two digits) or 3 (first three digits).

        inform: tells the number of registries that are being subjected to
            the Analysis; defaults to True

        digs: number of first digits to consider. Must be 1 (first digit),
            2 (first two digits) or 3 (first three digits).

        confidence: confidence level to draw lower and upper limits when
            plotting and to limit the top deviations to show. Defaults to None.

        high_Z: chooses which Z scores to be used when displaying results,
            according to the confidence level chosen. Defaluts to 'pos',
            which will highlight only values higher than the expexted
            frequencies; 'all' will highlight both extremes (positive and
            negative); and an integer, which will use the first n entries,
            positive and negative, regardless of whether Z is higher than
            the confidence or not.

        limit_N: sets a limit to N for the calculation of the Z score
            if the sample is too big. Defaults to None.

        MAD: calculates the Mean Absolute Difference between the
            found and the expected distributions; defaults to False.

        MSE: calculates the Mean Square Error of the sample; defaults to
            False.

        show_plot: draws the test plot. Defaults to True.

        ret_df: returns the test DataFrame. Defaults to False. True if run by
            the test function.
        '''
        # Check on the possible values for confidence lavels
        if str(confidence) not in list(confs.keys()):
            raise ValueError("Value of parameter -confidence- must be one\
 of the following: {0}".format(list(confs.keys())))
        # Check on possible digits
        if digs not in [1, 2, 3]:
            raise ValueError("The value assigned to the parameter -digs-\
 was {0}. Value must be 1, 2 or 3.".format(digs))

        # self[digs_dict[digs]] = self.ZN.astype(str).str[:digs].astype(int)
        temp = self.loc[self.ZN >= 10 ** (digs - 1)]
        temp[digs_dict[digs]] = (temp.ZN // 10 ** ((np.log10(temp.ZN).astype(
                                                   int)) - (digs - 1))).astype(
                                                       int)
        n, m = 10 ** (digs - 1), 10 ** (digs)
        x = np.arange(n, m)

        if simple:
            inform = False
            show_plot = False
            df = _prep_(temp, digs, limit_N=limit_N, simple=True,
                        confidence=None)
        else:
            N, df = _prep_(temp, digs, limit_N=limit_N, simple=False,
                           confidence=confidence)

        if inform:
            print("\nTest performed on {0} registries.\nDiscarded {1} \
records < {2} after preparation.".format(len(self), len(self) - len(temp),
                                         10 ** (digs - 1)))
            if confidence is not None:
                _inform_(df, high_Z=high_Z, conf=confs[str(confidence)])

        # Mean absolute difference
        if MAD:
            self.MAD = _mad_(df, test=digs, inform=inform)

        # Mean Square Error
        if MSE:
            self.MSE = _mse_(df, inform=inform)

        # Chi-square statistic
        if chi_square:
            self.chi_square = _chi_square_(df, ddf=len(df) - 1,
                                           confidence=confidence,
                                           inform=inform)
        # KS test
        if KS:
            self.KS = _KS_(df, confidence=confidence, N=len(temp),
                           inform=inform)

        # Plotting the expected frequncies (line) against the found ones(bars)
        if show_plot:
            _plot_dig_(df, x=x, y_Exp=df.Expected, y_Found=df.Found, N=N,
                       figsize=(2 * (digs ** 2 + 5), 1.5 * (digs ** 2 + 5)),
                       conf_Z=confs[str(confidence)])
        if ret_df:
            return df

    def second_digit(self, inform=True, confidence=None, high_Z='pos',
                     limit_N=None, MAD=False, MSE=False, chi_square=False,
                     KS=False, show_plot=True, simple=False, ret_df=False):
        '''
        Performs the Benford Second Digit test with the series of
        numbers provided.

        inform -> tells the number of registries that are being subjected to
            the Analysis; defaults to True

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

        limit_N: sets a limit to N for the calculation of the Z score
            if the sample is too big. Defaults to None.

        MSE: calculates the Mean Square Error of the sample; defaults to
            False.

        show_plot: draws the test plot.

        ret_df: returns the test DataFrame. Defaults to False. True if run by
            the test function.
        '''
        if str(confidence) not in list(confs.keys()):
            raise ValueError("Value of -confidence- must be one of the\
 following: {0}".format(list(confs.keys())))

        conf = confs[str(confidence)]

        # self['SD'] = self.ZN.astype(str).str[1:2].astype(int)
        temp = self.loc[self.ZN >= 10]
        temp['SD'] = (temp.ZN // 10**((np.log10(temp.ZN)).astype(
                      int) - 1)) % 10

        if simple:
            inform = False
            show_plot = False
            df = _prep_(temp, 22, limit_N=limit_N, simple=True,
                        confidence=None)
        else:
            N, df = _prep_(temp, 22, limit_N=limit_N, simple=False,
                           confidence=confidence)

        if inform:
            print("\nTest performed on {0} registries.\nDiscarded \
{1} records < 10 after preparation.".format(N, N - len(temp)))
            if confidence is not None:
                _inform_(df, high_Z, conf)

        # Mean absolute difference
        if MAD:
            self.MAD = _mad_(df, test=22, inform=inform)

        # Mean Square Error
        if MSE:
            self.MSE = _mse_(df, inform=inform)

        # Chi-square statistic
        if chi_square:
            self.chi_square = _chi_square_(df, ddf=9, confidence=confidence,
                                           inform=inform)
        # KS test
        if KS:
            self.KS = _KS_(df, confidence=confidence, N=len(temp),
                           inform=inform)

        # Plotting the expected frequncies (line) against the found ones(bars)
        if show_plot:
            _plot_dig_(df, x=np.arange(0, 10), y_Exp=df.Expected,
                       y_Found=df.Found, N=N, figsize=(10, 6), conf_Z=conf)
        if ret_df:
            return df

    def last_two_digits(self, inform=True, confidence=None, high_Z='pos',
                        limit_N=None, MAD=False, MSE=False, chi_square=False, KS=False,
                        show_plot=True, simple=False, ret_df=False):
        '''
        Performs the Benford Last Two Digits test with the series of
        numbers provided.

        inform -> tells the number of registries that are being subjected to
            the Analysis; defaults to True

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

        limit_N: sets a limit to N for the calculation of the Z score
            if the sample is too big. Defaults to None.

        MSE: calculates the Mean Square Error of the sample; defaults to
            False.

        show_plot: draws the test plot.

        '''
        if str(confidence) not in list(confs.keys()):
            raise ValueError("Value of -confidence- must be one of the \
following: {0}".format(list(confs.keys())))

        conf = confs[str(confidence)]

        temp = self.loc[self.ZN >= 1000]
        temp['L2D'] = temp.ZN % 100

        if simple:
            inform = False
            show_plot = False
            df = _prep_(temp, -2, limit_N=limit_N, simple=True,
                        confidence=None)
        else:
            N, df = _prep_(temp, -2, limit_N=limit_N, simple=False,
                           confidence=confidence)

        if inform:
            print("\nTest performed on {0} registries.\nDiscarded {1} \
records < 1000 after preparation".format(len(self), len(self) - len(temp)))
            if confidence is not None:
                _inform_(df, high_Z, conf)

        # Mean absolute difference
        if MAD:
            self.MAD = _mad_(df, test=-2, inform=inform)

        # Mean Square Error
        if MSE:
            self.MSE = _mse_(df, inform=inform)

        # Chi-square statistic
        if chi_square:
            self.chi_square = _chi_square_(df, ddf=99, confidence=confidence,
                                           inform=inform)
        # KS test
        if KS:
            self.KS = _KS_(df, confidence=confidence, N=len(temp),
                           inform=inform)

        # Plotting expected frequencies (line) versus found ones (bars)
        if show_plot:
            _plot_dig_(df, x=np.arange(0, 100), y_Exp=df.Expected,
                       y_Found=df.Found, N=N, figsize=(15, 8),
                       conf_Z=conf, text_x=True)
        if ret_df:
            return df

    def summation(self, digs=2, top=20, inform=True, show_plot=True,
                  ret_df=False):
        '''
        Performs the Summation test. In a Benford series, the sums of the
        entries begining with the same digits tends to be the same.

        digs -> tells the first digits to use. 1- first; 2- first two;
                3- first three. Defaults to 2.

        top -> choses how many top values to show. Defaults to 20.

        show_plot -> plots the results. Defaults to True.
        '''

        if digs not in [1, 2, 3]:
            raise ValueError("The value assigned to the parameter -digs-\
 was {0}. Value must be 1, 2 or 3.".format(digs))

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
        df.columns.values[1] = 'Sum'
        df = df[['Sum', 'Percent']]
        df['AbsDif'] = np.absolute(df.Percent - li)

        # Populate dict with the most relevant entries
        # self.maps[dig_name] = np.array(_inform_and_map_(s, inform,
        #                                high_Z=top, conf=None)).astype(int)
        if inform:
            # N = len(self)
            print("\nTest performed on {0} registries.\n".format(len(self)))
            print("The top {0} diferences are:\n".format(top))
            print(df[:top])

        if show_plot:
            _plot_sum_(df, figsize=(
                       2 * (digs ** 2 + 5), 1.5 * (digs ** 2 + 5)), li=li)

        if ret_df:
            return df

    def duplicates(self, inform=True, top_Rep=20):
        '''
        Performs a duplicates test and maps the duplicates count in descending
        order.

        inform -> tells how many duplicated entries were found and prints the
            top numbers according to the top_Rep parameter. Defaluts to True.

        top_Rep -> int or None. Chooses how many duplicated entries will be
            shown withe the top repititions. Defaluts to 20. If None, returns
            al the ordered repetitions.
        '''
        if top_Rep is not None and not isinstance(top_Rep, int):
            raise ValueError('The top_Rep parameter must be an int or None.')

        dup = self[['Seq']][self.Seq.duplicated(keep=False)]
        dup_count = dup.groupby(self.Seq).count()

        dup_count.index.names = ['Entries']
        dup_count.rename(columns={'Seq': 'Count'}, inplace=True)

        dup_count.sort_values('Count', ascending=False, inplace=True)

        self.maps['dup'] = dup_count.index[:top_Rep].values  # np.array

        if inform:
            print('Found {0} duplicated entries'.format(len(dup_count)))
            print('The entries with the {0} highest repitition counts are:'
                  .format(top_Rep))
            print(dup_count.head(top_Rep))
        else:
            return dup_count(top_Rep)


class Mantissas(pd.Series):
    '''
    Returns a Series with the data mantissas,

    Parameters
    ----------
    data: sequence to compute mantissas from, numpy 1D array, pandas
        Series of pandas DataFrame column.
    '''
    def __init__(self, data):
        if isinstance(data, np.ndarray):
            data = pd.Series(data)
        elif isinstance(data, pd.Series):
            pass
        else:
            raise ValueError('data must be a numpy array or a pandas Series')
        data.dropna(inplace=True)
        data = data.loc[data != 0]
        pd.Series.__init__(self, _getMantissas_(np.abs(data)))

        self.stats = {'Mean': self.mean(), 'Var': self.var(),
                      'Skew': self.skew(), 'Kurt': self.kurt()}

    def inform(self):
        print("The Mantissas MEAN is {0}. \t\tRef: 0.5.".
              format(self.stats['Mean']))
        print("The Mantissas VARIANCE is {0}. \tRef: 0.083333.".
              format(self.stats['Var']))
        print("The Mantissas SKEWNESS is {0}. \tRef: 0.".
              format(self.stats['Skew']))
        print("The Mantissas KURTOSIS is {0}. \tRef: -1.2.".
              format(self.stats['Kurt']))

    def show_plot(self, figsize=(15, 8)):
        '''
        plots the ordered mantissas and a line with the expected
                inclination. Defaults to True.

        figsize -> tuple that sets the figure size
        '''
        self.sort_values(inplace=True)
        x = np.arange(1, len(self) + 1)
        n = np.ones(len(self)) / len(self)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.plot(x, self, linestyle='--', color=colors['s'],
                linewidth=3, label='Mantissas')
        ax.plot(x, n.cumsum(), color=colors['m'],
                linewidth=2, label='Expected')
        plt.ylim((0, 1.))
        plt.xlim((1, len(self) + 1))
        ax.set_facecolor(colors['b'])
        plt.legend(loc='upper left')
        plt.show()


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
        If integers, set to 0. If set to -infer-, it will deal separately
        (and differently) with each registry.


    sign: tells which portion of the data to consider. pos: only the positive
        entries; neg: only negative entries; all: all entries but zeros.
        Defaults to all.
    '''
    def __init__(self, data, test, window, decimals=2, sign='all'):

        test = _check_test_(test)

        if not isinstance(data, Analysis):
            start = Analysis(data, sign=sign, decimals=decimals, inform=False)

        Exp, ind = _prep_to_roll_(start, test)

        pd.Series.__init__(self, start[digs_dict[test]].rolling(
            window=window).apply(_mad_to_roll_, args=(Exp, ind)))

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
        If integers, set to 0. If set to -infer-, it will deal separately
        (and differently) with each registry.


    sign: tells which portion of the data to consider. pos: only the positive
        entries; neg: only negative entries; all: all entries but zeros.
        Defaults to all.
    '''
    def __init__(self, data, test, window, decimals=2, sign='all'):

        test = _check_test_(test)

        if not isinstance(data, Analysis):
            start = Analysis(data, sign=sign, decimals=decimals, inform=False)

        Exp, ind = _prep_to_roll_(start, test)

        pd.Series.__init__(self, start[digs_dict[test]].rolling(
            window=window).apply(_mse_to_roll_, args=(Exp, ind)))

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


def _chi_square_(frame, ddf, confidence, inform=True):
    '''
    Returns the chi-square statistic of the found distributions and compares
    it with the critical chi-square of such a sample, according to the
    confidence level chosen and the degrees of freedom - len(sample) -1.

    Parameters
    ----------
    frame:      DataFrame with Foud, Expected and their difference columns.

    ddf:        Degrees of freedom to consider.

    confidence: Confidence level - confs dict.

    inform:     prints the chi-squre result and compares to the critical
    chi-square for the sample. Defaults to True.
    '''
    if confidence is None:
        print('\nChi-square test needs confidence other than None.')
        return
    else:
        exp_counts = frame.Counts.sum() * frame.Expected
        dif_counts = frame.Counts - exp_counts
        found_chi = (dif_counts ** 2 / exp_counts).sum()
        crit_chi = crit_chi2[ddf][str(confidence)]
        if inform:
            print("\nThe Chi-square statistic is {0}".format(found_chi))
            print("Critical Chi-square for this series: {0}".format(crit_chi))
        return (found_chi, crit_chi)


def _KS_(frame, confidence, N, inform=True):
    '''
    Returns the Kolmogorov-Smirnov test of the found distributions
    and compares it with the critical chi-square of such a sample,
    according to the confidence level chosen.

    Parameters
    ----------
    frame: DataFrame with Foud and Expected distributions.

    confidence: Confidence level - confs dict.

    N: Sample size

    inform: prints the KS result and the critical value for the sample.
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
        crit_KS = KS_crit[str(confidence)] / np.sqrt(N)

        if inform:
            print("\nThe Kolmogorov-Smirnov statistic is {0}".format(suprem))
            print("Critical K-S for this series: {0}".format(crit_KS))
        return (suprem, crit_KS)


def _mad_(frame, test, inform=True):
    '''
    Returns the Mean Absolute Deviation (MAD) between the found and the
    expected proportions.

    Parameters
    ----------

    frame: DataFrame with the Absolute Deviations already calculated.

    test: Test to compute the MAD from (F1D, SD, F2D...)

    inform: prints the MAD result and compares to limit values of
        conformity. Defaults to True.
    '''
    mad = frame.AbsDif.mean()

    if inform:
        print("\nThe Mean Absolute Deviation is {0}".format(mad))

        if test != -2:
            print("For the {0}:\n\
            - 0.0000 to {1}: Close Conformity\n\
            - {1} to {2}: Acceptable Conformity\n\
            - {2} to {3}: Marginally Acceptable Conformity\n\
            - Above {3}: Nonconformity".format(mad_dict[digs_dict[test]],
                                               mad_dict[test][0],
                                               mad_dict[test][1],
                                               mad_dict[test][2]))
        else:
            pass
    return mad


def _mse_(frame, inform=True):
    '''
    Returns the test's Mean Square Error

    frame -> DataFrame with the already computed Absolute Deviations between
            the found and expected proportions

    inform -> Prints the MSE. Defaults to True. If False, returns MSE.
    '''
    mse = (frame.AbsDif ** 2).mean()

    if inform:
        print("\nMean Square Error = {0}".format(mse))

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
        y_max = df.Expected.max() + (10 ** -(digs) / 3)
        fig, ax = plt.subplots(figsize=(2 * (digs ** 2 + 5), 1.5 *
                                        (digs ** 2 + 5)))
    elif digs == 22:
        y_max = .13
        fig, ax = plt.subplots(figsize=(14, 10.5))
    elif digs == -2:
        y_max = .011
        fig, ax = plt.subplots(figsize=(15, 8))
    plt.title('Expected Benford Distributions', size='xx-large')
    plt.xlabel(df.index.name, size='x-large')
    plt.ylabel('Distribution (%)', size='x-large')
    ax.set_facecolor(colors['b'])
    ax.set_ylim(0, y_max)
    ax.bar(df.index, df.Expected, color=colors['t'])
    ax.set_xticks(df.index)
    ax.set_xticklabels(df.index)
    plt.show()


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
    fig, ax = plt.subplots(figsize=figsize)
    plt.title('Expected vs. Found Distributions', size='xx-large')
    plt.xlabel('Digits', size='x-large')
    plt.ylabel('Distribution (%)', size='x-large')
    bars = plt.bar(x, y_Found * 100., color=colors['m'],
                   label='Found', zorder=3)
    ax.set_xticks(x + .4)
    ax.set_xticklabels(x)
    ax.plot(x, y_Exp * 100., color=colors['s'], linewidth=2.5,
            label='Benford', zorder=4)
    # ax.grid(axis='y', color='w', linestyle='-', zorder=0)
    ax.set_facecolor(colors['b'])
    if text_x:
        ind = np.array(df.index).astype(str)
        ind[:10] = np.array(['00', '01', '02', '03', '04', '05',
                             '06', '07', '08', '09'])
        plt.xticks(x, ind, rotation='vertical')
    # Plotting the Upper and Lower bounds considering the Z for the
    # informed confidence level
    ax.legend()
    if conf_Z is not None:
        sig = conf_Z * np.sqrt(y_Exp * (1 - y_Exp) / N)
        upper = y_Exp + sig + (1 / (2 * N))
        lower = y_Exp - sig - (1 / (2 * N))
        u = (y_Found < lower) | (y_Found > upper)
        for i, b in enumerate(bars):
            if u.iloc[i]:
                b.set_color(colors['af'])
        lower *= 100.
        upper *= 100.
        ax.plot(x, upper, color=colors['s'], zorder=5)
        ax.plot(x, lower, color=colors['s'], zorder=5)
        ax.fill_between(x, upper, lower, color=colors['s'],
                        alpha=.3, label='Conf')
    plt.show()


def _plot_sum_(df, figsize, li):
    '''
    Plotss the summation test results

    df -> DataFrame with the data to be plotted

    figsize - > tuple to state the size of the plot figure

    li -> values with which to draw the horizontal line
    '''
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    plt.title('Expected vs. Found Sums')
    plt.xlabel('Digits')
    plt.ylabel('Sums')
    ax.bar(df.index, df.Percent, color=colors['m'],
           label='Found Sums', zorder=3)
    ax.axhline(li, color=colors['s'], linewidth=2, label='Expected', zorder=4)
    ax.set_facecolor(colors['b'])
    ax.legend()


def _set_N_(len_df, limit_N):
    # Assigning to N the superior limit or the lenght of the series
    if limit_N is None or limit_N > len_df:
        N = len_df
    # Check on limit_N being a positive integer
    else:
        if limit_N < 0 or not isinstance(limit_N, int):
            raise ValueError("-limit_N- must be None or a positive \
integer.")
        else:
            N = limit_N
    return N


def _base_(digs):
    '''
    Returns the base instance for the proper test to be performed
    depending on the digit
    '''
    if digs == 1:
        return First(1, plot=False)
    elif digs == 2:
        return First(2, plot=False)
    elif digs == 3:
        return First(3, plot=False)
    elif digs == 22:
        return Second(plot=False)
    else:
        return LastTwo(num=True, plot=False)


def _prep_(df, digs, limit_N, simple=False, confidence=None):
    '''
    Transforms the original number sequence into a DataFrame reduced
    by the ocurrences of the chosen digits, creating other computed
    columns
    '''
    N = _set_N_(len(df), limit_N=limit_N)

    col = digs_dict[digs]

    # get the number of occurrences of the last two digits
    v = df[col].value_counts()
    # get their relative frequencies
    p = df[col].value_counts(normalize=True)
    # crate dataframe from them
    dd = pd.DataFrame({'Counts': v, 'Found': p}).sort_index()
    # join the dataframe with the one of expected Benford's frequencies
    dd = _base_(digs).join(dd).fillna(0)
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


def first_digits(data, digs, decimals=2, sign='all', inform=True,
                 confidence=None, high_Z='pos', limit_N=None,
                 MAD=False, MSE=False, chi_square=False, KS=False,
                 show_plot=True):
    '''
    Performs the Benford First Digits test on the series of
    numbers provided.

    Parameters
    ----------

    data: sequence of numbers to be evaluated. Must be a numpy 1D array,
        a pandas Series or a pandas DataFrame column, with values being
        integers or floats.

    decimals: number of decimal places to consider. Defaluts to 2.
        If integers, set to 0. If set to -infer-, it will deal separately
        (and differently) with each registry.

    sign: tells which portion of the data to consider. pos: only the positive
        entries; neg: only negative entries; all: all entries but zeros.
        Defaults to all.`

    digs: number of first digits to consider. Must be 1 (first digit),
        2 (first two digits) or 3 (first three digits).

    inform: tells the number of registries that are being subjected to
        the Analysis and returns tha analysis DataFrame sorted by the
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

    limit_N: sets a limit to N for the calculation of the Z score
        if the sample is too big. Defaults to None.

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
    if not isinstance(data, Analysis):
        data = Analysis(data, decimals=decimals, sign=sign, inform=inform)

    data = data.first_digits(digs, inform=inform, confidence=confidence,
                             high_Z=high_Z, limit_N=limit_N, MAD=MAD, MSE=MSE,
                             chi_square=chi_square, KS=KS, show_plot=show_plot,
                             ret_df=True)

    if confidence is not None:
        data = data[['Counts', 'Found', 'Expected', 'Z_score']]
        return data.sort_values('Z_score', ascending=False)
    else:
        return data[['Counts', 'Found', 'Expected']]


def second_digit(data, decimals=2, sign='all', inform=True,
                 confidence=None, high_Z='pos', limit_N=None,
                 MAD=False, MSE=False, chi_square=False, KS=False,
                 show_plot=True):
    '''
    Performs the Benford Second Digits test on the series of
    numbers provided.

    Parameters
    ----------

    data: sequence of numbers to be evaluated. Must be a numpy 1D array,
        a pandas Series or a pandas DataFrame column, with values being
        integers or floats.

    decimals: number of decimal places to consider. Defaluts to 2.
        If integers, set to 0. If set to -infer-, it will deal separately
        (and differently) with each registry.

    sign: tells which portion of the data to consider. pos: only the positive
        entries; neg: only negative entries; all: all entries but zeros.
        Defaults to all.`

    inform: tells the number of registries that are being subjected to
        the Analysis and returns tha analysis DataFrame sorted by the
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

    limit_N: sets a limit to N for the calculation of the Z score
        if the sample is too big. Defaults to None.

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
    if not isinstance(data, Analysis):
        data = Analysis(data, sign=sign, decimals=decimals, inform=inform)

    data = data.second_digit(inform=inform, confidence=confidence,
                             high_Z=high_Z, limit_N=limit_N, MAD=MAD, MSE=MSE,
                             chi_square=chi_square, KS=KS, show_plot=show_plot,
                             ret_df=True)
    if confidence is not None:
        data = data[['Counts', 'Found', 'Expected', 'Z_score']]
        return data.sort_values('Z_score', ascending=False)
    else:
        return data[['Counts', 'Found', 'Expected']]


def last_two_digits(data, decimals=2, sign='all', inform=True,
                    confidence=None, high_Z='pos', limit_N=None,
                    MAD=False, MSE=False, chi_square=False, KS=False,
                    show_plot=True):
    '''
    Performs the Last Two Digits test on the series of
    numbers provided.

    Parameters
    ----------

    data: sequence of numbers to be evaluated. Must be a numpy 1D array,
        a pandas Series or a pandas DataFrame column,with values being
        integers or floats.

    decimals: number of decimal places to consider. Defaluts to 2.
        If integers, set to 0. If set to -infer-, it will deal separately
        (and differently) with each registry.

    sign: tells which portion of the data to consider. pos: only the positive
        entries; neg: only negative entries; all: all entries but zeros.
        Defaults to all.`

    inform: tells the number of registries that are being subjected to
        the Analysis and returns tha analysis DataFrame sorted by the
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

    limit_N: sets a limit to N for the calculation of the Z score
        if the sample is too big. Defaults to None.

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
    if not isinstance(data, Analysis):
        data = Analysis(data, decimals=decimals, sign=sign, inform=inform)

    data = data.last_two_digits(inform=inform, confidence=confidence,
                                high_Z=high_Z, limit_N=limit_N, MAD=MAD,
                                MSE=MSE, chi_square=chi_square, KS=KS,
                                show_plot=show_plot, ret_df=True)

    if confidence is not None:
        data = data[['Counts', 'Found', 'Expected', 'Z_score']]
        return data.sort_values('Z_score', ascending=False)
    else:
        return data[['Counts', 'Found', 'Expected']]


def mantissas(data, inform=True, show_plot=True):
    '''
    Returns a Series with the data mantissas,

    Parameters
    ----------
    data: sequence to compute mantissas from, numpy 1D array, pandas
        Series of pandas DataFrame column.

    inform: prints the mamtissas mean, variance, skewness and kurtosis
        for the sequence studied, along with reference values.

    show_plot: plots the ordered mantissas and a line with the expected
        inclination. Defaults to True.
    '''
    mant = Mantissas(data)
    if inform:
        mant.inform()
    if show_plot:
        mant.show_plot()
    return mant


def summation(data, digs=2, decimals=2, sign='all', top=20, inform=True,
              show_plot=True):
    '''
    Performs the Summation test. In a Benford series, the sums of the
    entries begining with the same digits tends to be the same.
    Works only with the First Digits (1, 2 or 3) test.

    Parameters
    ----------

    digs: tells the first digits to use: 1- first; 2- first two;
        3- first three. Defaults to 2.

    decimals: number of decimal places to consider. Defaluts to 2.
        If integers, set to 0. If set to -infer-, it will deal separately
        (and differently) with each registry.

    top: choses how many top values to show. Defaults to 20.

    show_plot: plots the results. Defaults to True.

    '''
    if not isinstance(data, Analysis):
        data = Analysis(data, sign=sign, decimals=decimals, inform=inform)

    data = data.summation(digs=digs, top=top, inform=inform,
                          show_plot=show_plot, ret_df=True)
    if inform:
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

    decimals: number of decimal places to consider. Defaluts to 2.
        If integers, set to 0. If set to -infer-, it will deal separately
        (and differently) with each registry.

    sign: tells which portion of the data to consider. pos: only the positive
        entries; neg: only negative entries; all: all entries but zeros.
        Defaults to all.`

    '''
    test = _check_test_(test)

    start = Analysis(data, sign=sign, decimals=decimals, inform=False)

    if test in [1, 2, 3]:
        start.first_digits(digs=test, inform=False, MAD=True, simple=True)
    elif test == 22:
        start.second_digit(inform=False, MAD=True, simple=True)
    else:
        start.last_two_digits(inform=False, MAD=True, simple=True)

    return start.MAD


def mse(data, test, decimals=2, sign='all'):
    '''
    Returns the Mean Squared Error of the Series
    '''
    test = _check_test_(test)
    start = Analysis(data, sign=sign, decimals=decimals, inform=False)
    if test in [1, 2, 3]:
        start.first_digits(digs=test, MAD=False, MSE=True, simple=True)
    elif test == 22:
        start.second_digit(MAD=False, MSE=True, simple=True)
    else:
        start.last_two_digits(MAD=False, MSE=True, simple=True)
    return start.MSE


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
        If integers, set to 0. If set to -infer-, it will deal separately
        (and differently) with each registry.

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
        If integers, set to 0. If set to -infer-, it will deal separately
        (and differently) with each registry.

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


def duplicates(data, top_Rep=20, inform=True):
    '''
    Performs a duplicates test and maps the duplicates count in descending
    order.

    Parameters
    ----------
    data: sequence to take the duplicates from. pandas Series or
        numpy Ndarray.

    inform: tells how many duplicated entries were found and prints the
        top numbers according to the top_Rep parameter. Defaluts to True.

    top_Rep: chooses how many duplicated entries will be
        shown withe the top repititions. int or None. Defaluts to 20.
        If None, returns al the ordered repetitions.
    '''
    if top_Rep is not None and not isinstance(top_Rep, int):
        raise ValueError('The top_Rep parameter must be an int or None.')

    if not isinstance(data, pd.Series):
        try:
            data = pd.Series(data)
        except ValueError:
            print('data must be a numpy Ndarray or a pandas Series.')

    dup = data.loc[data.duplicated(keep=False)]
    dup_count = dup.value_counts()

    dup_count.index.names = ['Entries']
    dup_count.name = 'Count'

    if inform:
        print('Found {0} duplicated entries'.format(len(dup_count)))
        print('The entries with the {0} highest repitition counts are:'
              .format(top_Rep))
        print(dup_count.head(top_Rep))

    return dup_count


def _subtract_sorted_(data):
    '''
    Subtracts the sorted sequence elements from each other, discarding zeros.
    Used in the Second Order test
    '''
    data.sort_values(inplace=True)
    data = data - data.shift(1)
    data = data.loc[data != 0]
    return data


def second_order(data, test, decimals=2, sign='all', inform=True, MAD=False,
                 confidence=None, high_Z='pos', limit_N=None, MSE=False,
                 show_plot=True):
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
        If integers, set to 0. If set to -infer-, it will deal separately
        (and differently) with each registry.

    sign: tells which portion of the data to consider. pos: only the positive
        entries; neg: only negative entries; all: all entries but zeros.
        Defaults to all.`

    inform: tells the number of registries that are being subjected to
        the Analysis and returns tha analysis DataFrame sorted by the
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

    limit_N: sets a limit to N for the calculation of the Z score
        if the sample is too big. Defaults to None.

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

    # if not isinstance(data, Analysis):
    data = Analysis(data, decimals=decimals, sign=sign,
                    sec_order=True, inform=inform)
    if test in [1, 2, 3]:
        data.first_digits(digs=test, inform=inform, MAD=MAD,
                          confidence=confidence, high_Z=high_Z,
                          limit_N=limit_N, MSE=MSE, show_plot=show_plot)
    elif test == 22:
        data.second_digit(inform=inform, MAD=MAD, confidence=confidence,
                          high_Z=high_Z, limit_N=limit_N, MSE=MSE,
                          show_plot=show_plot)
    else:
        data.last_two_digits(inform=inform, MAD=MAD,
                             confidence=confidence, high_Z=high_Z,
                             limit_N=limit_N, MSE=MSE, show_plot=show_plot)
    return data


def _check_test_(test):
    '''
    Checks the test chosen, both for int or str values
    '''
    if isinstance(test, int):
        if test in digs_dict.keys():
            return test
        else:
            raise ValueError('test was set to {0}. Should be one of {1}'
                             .format(test, digs_dict.keys()))
    elif isinstance(test, str):
        if test in rev_digs.keys():
            return rev_digs[test]
        else:
            raise ValueError('test was set to {0}. Should be one of {1}'
                             .format(test, rev_digs.keys()))
    else:
        raise ValueError('Wrong value chosen for test parameter. \
Please, refer to docs.')


def _inform_(df, high_Z, conf):
    '''
    Selects and sorts by the Z_stats chosen to be considered, informing or not,
    and populating the maps dict for further back analysis of the entries.
    '''

    if isinstance(high_Z, int):
        if conf is not None:
            dd = df[['Expected', 'Found', 'Z_score'
                     ]].sort_values('Z_score', ascending=False
                                    ).head(high_Z)
            print('\nThe entries with the top {0} Z scores are\
:\n'.format(high_Z))
        # Summation Test
        else:
            dd = df.sort_values('AbsDif', ascending=False
                                ).head(high_Z)
            print('\nThe entries with the top {0} absolute deviations \
are:\n'.format(high_Z))
    else:
        if high_Z == 'pos':
            m1 = df.Dif > 0
            m2 = df.Z_score > conf
            dd = df[['Expected', 'Found', 'Z_score'
                     ]].loc[m1 & m2].sort_values('Z_score', ascending=False)
            print('\nThe entries with the significant positive deviations \
are:\n')
        elif high_Z == 'neg':
            m1 = df.Dif < 0
            m2 = df.Z_score > conf
            dd = df[['Expected', 'Found', 'Z_score'
                     ]].loc[m1 & m2].sort_values('Z_score', ascending=False)
            print('\nThe entries with the significant negative deviations \
are:\n')
        else:
            dd = df[['Expected', 'Found', 'Z_score'
                     ]].loc[df.Z_score > conf].sort_values('Z_score',
                                                           ascending=False)
            print('\nThe entries with the significant deviations are:\n')
    print(dd)

# to do:

# XXXXXXX MAPPING BACK XXXXXXX