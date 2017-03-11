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

# Imports
from __future__ import print_function
from __future__ import division


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


digs_dict = {1: 'F1D', 2: 'F2D', 3: 'F3D', 22: 'SD', -2: 'L2D'}

colors = {'m': '#00798c', 'b': '#E2DCD8', 's': '#9c3848',
          'af': '#edae49', 'ab': '#33658a', 'h': '#d1495b',
          'h2': '#f64740', 't': '#16DB93'}


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
            p = self.plot(kind='bar', color=colors['t'],
                          figsize=(2 * (digs ** 2 + 5), 1.5 * (digs ** 2 + 5)))
            p.set_axis_bgcolor(colors['b'])


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
            p = self.plot(kind='bar', color=colors['t'],
                          figsize=(14, 10.5), ylim=(0, .14))
            p.set_axis_bgcolor(colors['b'])


class LastTwo(pd.DataFrame):
    '''
    Returns the expected probabilities of the Last Two Digits
    according to Benford's distribution.

    plot: option to plot a bar chart of the Expected proportions.
        Defaults to True.
    '''
    def __init__(self, plot=True):
        exp = np.array([1 / 99.] * 100)
        pd.DataFrame.__init__(self, {'Expected': exp, 'Last_2_Dig': _lt_()})
        self.set_index('Last_2_Dig', inplace=True)
        if plot:
            p = self.plot(kind='bar', figsize=(15, 8), color=colors['t'],
                          ylim=(0, 0.013))
            p.set_axis_bgcolor(colors['b'])


class Analysis(pd.DataFrame):
    '''
    Prepares the data for Analysis. pandas DataFrame subclass.

    Parameters
    ----------

    data: sequence of numbers to be evaluated. Must be a numpy 1D array,
        a pandas Series or a pandas DataFrame column, with values being
            integers or floats.

    dec: number of decimal places to consider. Defaluts to 2.
        If integers, set to 0.

    sec_order: choice for the Second Order Test, which cumputes the
        differences between the ordered entries before running the Tests.

    inform: tells the number of registries that are being subjected to
        the Analysis; defaults to True
    '''
    maps = {}  # dict for recording the indexes to be mapped back to the
    # dict of confidence levels for further use
    stats = {}
    confs = {'None': None, '80': 1.285, '85': 1.435, '90': 1.645, '95': 1.96,
             '99': 2.576, '99.9': 3.29, '99.99': 3.89, '99.999': 4.417,
             '99.9999': 4.892, '99.99999': 5.327}

    def __init__(self, data, sign='all', dec=2, sec_order=False, inform=True):
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
            self.sort_values('Seq', inplace=True)
            self.drop_duplicates(inplace=True)
            self.Seq = self.Seq - self.Seq.shift(1)
            self.dropna(inplace=True)
            if inform:
                print('Second Order Test. Initial series reduced to {0}\
 entries.'.format(len(self)))

        self['ZN'] = np.abs(self.Seq * (10**dec)).astype(int)  # dec - decimals

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
        p = p[p.Seq > 0].sort_values('Mant')
        print("The Mantissas MEAN is {0}. Ref: 0.5.".format(p.Mant.mean()))
        print("The Mantissas VARIANCE is {0}. Ref: 0.083333.".format(
              p.Mant.var()))
        print("The Mantissas SKEWNESS is {0}. \tRef: 0.".format(p.Mant.skew()))
        print("The Mantissas KURTOSIS is {0}. \tRef: -1.2.".
              format(p.Mant.kurt()))
        N = len(p)

        if plot:
            p['x'] = np.arange(1, N + 1)
            n = np.ones(N) / N
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            ax.plot(p.x, p.Mant, 'r-', p.x, n.cumsum(), 'b--',
                    linewidth=2)
            plt.ylim((0, 1.))
            plt.xlim((1, N + 1))
            plt.show()

    def first_digits(self, digs, inform=True, MAD=True, conf_level=95,
                     high_Z='pos', limit_N=None, MSE=False, show_plot=True,
                     simple=False, ret_df=False):
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

        MAD: calculates the Mean Absolute Difference between the
            found and the expected distributions; defaults to True.

        conf_level: confidence level to draw lower and upper limits when
            plotting and to limit the top deviations to show. Defaults to 95.
            If None, no boundaries will be drawn.

        high_Z: chooses which Z scores to be used when displaying results,
            according to the confidence level chosen. Defaluts to 'pos',
            which will highlight only values higher than the expexted
            frequencies; 'all' will highlight both extremes (positive and
            negative); and an integer, which will use the first n entries,
            positive and negative, regardless of whether Z is higher than
            the conf_level or not.

        limit_N: sets a limit to N for the calculation of the Z score
            if the sample is too big. Defaults to None.

        MSE: calculates the Mean Square Error of the sample; defaults to
            False.

        show_plot: draws the test plot.

        ret_df: returns the test DataFrame. Defaults to False. True if run by
            the test function.
        '''
        # Check on the possible values for confidence lavels
        if str(conf_level) not in list(self.confs.keys()):
            raise ValueError("Value of parameter -conf_level- must be one\
 of the following: {0}".format(list(self.confs.keys())))
        # Check on possible digits
        if digs not in [1, 2, 3]:
            raise ValueError("The value assigned to the parameter -digs-\
 was {0}. Value must be 1, 2 or 3.".format(digs))

        self[digs_dict[digs]] = self.ZN.astype(str).str[:digs].astype(int)

        temp = self.loc[self.ZN >= 10 ** (digs - 1)]

        n, m = 10 ** (digs - 1), 10 ** (digs)
        x = np.arange(n, m)

        if simple:
            inform = False
            show_plot = False
            N, df = _simple_prep_(temp, digs, limit_N=limit_N)
        else:
            N, df = _prep_(temp, digs, limit_N=limit_N)

        if inform:
            print("\nTest performed on {0} registries.\nDiscarded {1} \
records < {2} after preparation.".format(len(self), len(self) - len(temp),
                                         10 ** (digs - 1)))
            _inform_(df, high_Z=high_Z, conf=self.confs[str(conf_level)])

        # Mean absolute difference
        if MAD:
            self.MAD = _mad_(df, test=digs_dict[digs], inform=inform)

        # Mean Square Error
        if MSE:
            self.MSE = _mse_(df, inform=inform)

        # Plotting the expected frequncies (line) against the found ones(bars)
        if show_plot:
            _plot_dig_(df, x=x, y_Exp=df.Expected, y_Found=df.Found, N=N,
                       figsize=(2 * (digs ** 2 + 5), 1.5 * (digs ** 2 + 5)),
                       conf_Z=self.confs[str(conf_level)])
        if ret_df:
            return df

    def second_digit(self, inform=True, MAD=True, conf_level=95,
                     MSE=False, high_Z='pos', limit_N=None,
                     show_plot=True, simple=False, ret_df=False):
        '''
        Performs the Benford Second Digit test with the series of
        numbers provided.

        inform -> tells the number of registries that are being subjected to
            the Analysis; defaults to True

        MAD: calculates the Mean Absolute Difference between the
            found and the expected distributions; defaults to True.

        conf_level: confidence level to draw lower and upper limits when
            plotting and to limit the top deviations to show. Defaults to 95.
            If None, no boundaries will be drawn.

        high_Z: chooses which Z scores to be used when displaying results,
            according to the confidence level chosen. Defaluts to 'pos',
            which will highlight only values higher than the expexted
            frequencies; 'all' will highlight both extremes (positive and
            negative); and an integer, which will use the first n entries,
            positive and negative, regardless of whether Z is higher than
            the conf_level or not.

        limit_N: sets a limit to N for the calculation of the Z score
            if the sample is too big. Defaults to None.

        MSE: calculates the Mean Square Error of the sample; defaults to
            False.

        show_plot: draws the test plot.

        ret_df: returns the test DataFrame. Defaults to False. True if run by
            the test function.
        '''
        if str(conf_level) not in list(self.confs.keys()):
            raise ValueError("Value of -conf_level- must be one of the\
 following: {0}".format(list(self.confs.keys())))

        conf = self.confs[str(conf_level)]

        self['SD'] = self.ZN.astype(str).str[1:2].astype(int)
        # self['SD'] = _create_dig_col_(self.ZN, 22)

        temp = self.loc[self.ZN >= 10]

        if simple:
            inform = False
            show_plot = False
            N, df = _simple_prep_(temp, 22, limit_N=limit_N)
        else:
            N, df = _prep_(temp, 22, limit_N=limit_N)

        if inform:
            print("\nTest performed on {0} registries.\nDiscarded \
{1} records < 10 after preparation.".format(N, N - len(temp)))
            _inform_(df, high_Z, conf)

        # Mean absolute difference
        if MAD:
            self.MAD = _mad_(df, 'SD', inform=inform)

        # Mean Square Error
        if MSE:
            self.MSE = _mse_(df, inform=inform)

        # Plotting the expected frequncies (line) against the found ones(bars)
        if show_plot:
            _plot_dig_(df, x=np.arange(0, 10), y_Exp=df.Expected,
                       y_Found=df.Found, N=N, figsize=(10, 6), conf_Z=conf)
        if ret_df:
            return df

    def last_two_digits(self, inform=True, MAD=False, conf_level=95,
                        high_Z='pos', limit_N=None, MSE=False,
                        show_plot=True, simple=False, ret_df=False):
        '''
        Performs the Benford Last Two Digits test with the series of
        numbers provided.

        inform -> tells the number of registries that are being subjected to
            the Analysis; defaults to True

        MAD: calculates the Mean Absolute Difference between the
            found and the expected distributions; defaults to True.

        conf_level: confidence level to draw lower and upper limits when
            plotting and to limit the top deviations to show. Defaults to 95.
            If None, no boundaries will be drawn.

        high_Z: chooses which Z scores to be used when displaying results,
            according to the confidence level chosen. Defaluts to 'pos',
            which will highlight only values higher than the expexted
            frequencies; 'all' will highlight both extremes (positive and
            negative); and an integer, which will use the first n entries,
            positive and negative, regardless of whether Z is higher than
            the conf_level or not.

        limit_N: sets a limit to N for the calculation of the Z score
            if the sample is too big. Defaults to None.

        MSE: calculates the Mean Square Error of the sample; defaults to
            False.

        show_plot: draws the test plot.

        '''
        if str(conf_level) not in list(self.confs.keys()):
            raise ValueError("Value of -conf_level- must be one of the \
following: {0}".format(list(self.confs.keys())))

        conf = self.confs[str(conf_level)]

        self['L2D'] = self.ZN.astype(str).str[-2:]

        temp = self.loc[self.ZN >= 1000]

        if simple:
            inform = False
            show_plot = False
            N, df = _simple_prep_(temp, -2, limit_N=limit_N)
        else:
            N, df = _prep_(temp, -2, limit_N=limit_N)

        if inform:
            print("\nTest performed on {0} registries.\nDiscarded {1} \
records < 1000 after preparation".format(len(self), len(self) - len(temp)))
            _inform_(df, high_Z, conf)

        # Mean absolute difference
        if MAD:
            self.MAD = _mad_(df, test='L2D', inform=inform)

        # Mean Square Error
        if MSE:
            self.MSE = _mse_(df, inform=inform)

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
        # Set the future dict key
        # dig_name = 'SUM{0}'.format(digs)
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
            raise ValueError('input must be a numpy array or a pandas Series')
        data.dropna(inplace=True)
        data = data.loc[data != 0]
        pd.Series.__init__(self, _getMantissas_(np.abs(data)))

    def mean(self):
        self.mean()

    def var(self):
        self.var()

    def skew(self):
        self.skew()

    def kurt(self):
        self.kurt()

    def inform(self):
        print("The Mantissas MEAN is {0}. \t\tRef: 0.5.".format(self.mean()))
        print("The Mantissas VARIANCE is {0}. \tRef: 0.083333.".
              format(self.var()))
        print("The Mantissas SKEWNESS is {0}. \tRef: 0.".format(self.skew()))
        print("The Mantissas KURTOSIS is {0}. \tRef: -1.2.".
              format(self.kurt()))

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
        ax.plot(x, self, linestyle='--', color=colors['s'], linewidth=3)
        ax.plot(x, n.cumsum(), color=colors['m'], linewidth=2)
        plt.ylim((0, 1.))
        plt.xlim((1, len(self) + 1))
        ax.set_axis_bgcolor(colors['b'])
        plt.show()


def _Z_test(frame, N):
    '''
    Returns the Z statistics for the proportions assessed

    frame -> DataFrame with the expected proportions and the already calculated
            Absolute Diferences between the found and expeccted proportions
    N -> sample size
    '''
    return (frame.AbsDif - (1 / (2 * N))) / np.sqrt(
           (frame.Expected * (1. - frame.Expected)) / N)


def _mad_(frame, test, inform=True):
    '''
    Returns the Mean Absolute Deviation (MAD) between the found and the
    expected proportions.

    Parameters
    ----------

    frame: DataFrame with the Absolute Deviations already calculated.
    test: Teste applied (F1D, SD, F2D...)

    inform: prints the MAD result and compares to limit values of
        conformity. Defaults to True. If False, returns the value.
    '''
    mad = frame.AbsDif.mean()

    if inform:
        if test[0] == 'F':
            if test == 'F1D':
                margins = ['0.006', '0.012', '0.015', 'Digit']
            elif test == 'F2D':
                margins = ['0.0012', '0.0018', '0.0022', 'Two Digits']
            else:
                margins = ['0.00036', '0.00044', '0.00050', 'Three Digits']
            print("\nThe Mean Absolute Deviation is {0}\n\
        For the First {1}:\n\
        - 0.0000 to {2}: Close Conformity\n\
        - {2} to {3}: Acceptable Conformity\n\
        - {3} to {4}: Marginally Acceptable Conformity\n\
        - Above {4}: Nonconformity".format(mad, margins[3], margins[0],
                                           margins[1], margins[2]))

        elif test == 'SD':
            print("\nThe Mean Absolute Deviation is {0}\n\
        For the Second Digits:\n\
        - 0.000 to 0.008: Close Conformity\n\
        - 0.008 to 0.010: Acceptable Conformity\n\
        - 0.010 to 0.012: Marginally Acceptable Conformity\n\
        - Above 0.012: Nonconformity".format(mad))

        else:
            print("\nThe Mean Absolute Deviation is {0}.\n".format(mad))

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
    The mantissa is the non-integer part of the log of a number.
    This fuction uses the element-wise array operations of numpy
    to get the mantissas of each number's log.

    arr: np.array of integers or floats ---> np.array of floats
    '''
    log_a = np.abs(np.log10(arr))
    return log_a - log_a.astype(int)  # the number - its integer part


def _lt_():
    '''
    Creates an array with strings of the possible last two digits
    '''
    n = np.arange(0, 100).astype(str)
    n[:10] = np.array(['00', '01', '02', '03', '04', '05',
                       '06', '07', '08', '09'])
    return n


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
    # y_Exp *= 100.
    # y_Found *= 100.
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
    ax.set_axis_bgcolor(colors['b'])
    if text_x:
        plt.xticks(x, df.index, rotation='vertical')
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
    l -> values with which to draw the horizontal line
    '''
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    plt.title('Expected vs. Found Sums')
    plt.xlabel('Digits')
    plt.ylabel('Sums')
    ax.bar(df.index, df.Percent, color=colors['m'],
           label='Found Sums', zorder=3)
    ax.axhline(li, color=colors['s'], linewidth=2, label='Expected', zorder=4)
    ax.set_axis_bgcolor(colors['b'])
    # ax.grid(axis='y', color='w', linestyle='-', zorder=0)
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
        return LastTwo(plot=False)


def _prep_(df, digs, limit_N):
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
    dd = _base_(digs).join(dd)
    # create column with absolute differences
    dd['Dif'] = dd.Found - dd.Expected
    dd['AbsDif'] = np.absolute(dd.Dif)
    # calculate the Z-test column an display the dataframe by descending
    # Z test
    dd['Z_test'] = _Z_test(dd, N)
    return N, dd


def _simple_prep_(df, digs, limit_N):
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
    dd = _base_(digs).join(dd)
    # create column with absolute differences
    dd['AbsDif'] = np.absolute(dd.Found - dd.Expected)
    return N, dd


def first_digits(data, digs, sign='all', dec=2, inform=True,
                 MAD=True, conf_level=95, high_Z='pos',
                 limit_N=None, MSE=False, show_plot=True):
    '''
    Performs the Benford First Digits test on the series of
    numbers provided.

    Parameters
    ----------

    data: sequence of numbers to be evaluated. Must be a numpy 1D array,
        a pandas Series or a pandas DataFrame column, with values being
        integers or floats.

    dec: number of decimal places to consider. Defaluts to 2.
        If integers, set to 0.

    inform: tells the number of registries that are being subjected to
        the Analysis; defaults to True

    digs: number of first digits to consider. Must be 1 (first digit),
        2 (first two digits) or 3 (first three digits).

    inform: tells the number of registries that are being subjected to
        the Analysis and returns tha analysis DataFrame sorted by the
        highest Z score down. Defaults to True.

    MAD: calculates the Mean Absolute Difference between the
        found and the expected distributions; defaults to True.

    conf_level: confidence level to draw lower and upper limits when
        plotting and to limit the top deviations to show. Defaults to 95.
        If None, no boundaries will be drawn.

    high_Z: chooses which Z scores to be used when displaying results,
        according to the confidence level chosen. Defaluts to 'pos',
        which will highlight only values higher than the expexted
        frequencies; 'all' will highlight both extremes (positive and
        negative); and an integer, which will use the first n entries,
        positive and negative, regardless of whether Z is higher than
        the conf_level or not.

    limit_N: sets a limit to N for the calculation of the Z score
        if the sample is too big. Defaults to None.

    MSE: calculates the Mean Square Error of the sample; defaults to
        False.

    show_plot: draws the test plot.

    '''
    if not isinstance(data, Analysis):
        data = Analysis(data, sign=sign, dec=dec, inform=inform)

    data = data.first_digits(digs, inform=inform, MAD=MAD,
                             conf_level=conf_level, high_Z=high_Z,
                             limit_N=limit_N, MSE=MSE,
                             show_plot=show_plot, ret_df=True)
    if inform:
        return data.sort_values('Z_test', ascending=False)
    else:
        return data


def second_digit(data, sign='all', dec=2, inform=True,
                 MAD=True, conf_level=95, high_Z='pos', limit_N=None,
                 MSE=False, show_plot=True):
    '''
    Performs the Benford Second Digits test on the series of
    numbers provided.

    Parameters
    ----------

    data: sequence of numbers to be evaluated. Must be a numpy 1D array,
        a pandas Series or a pandas DataFrame column, with values being
        integers or floats.

    dec: number of decimal places to consider. Defaluts to 2.
        If integers, set to 0.

    inform: tells the number of registries that are being subjected to
        the Analysis and returns tha analysis DataFrame sorted by the
        highest Z score down. Defaults to True.

    MAD: calculates the Mean Absolute Difference between the
        found and the expected distributions; defaults to True.

    conf_level: confidence level to draw lower and upper limits when
        plotting and to limit the top deviations to show. Defaults to 95.
        If None, no boundaries will be drawn.

    high_Z: chooses which Z scores to be used when displaying results,
        according to the confidence level chosen. Defaluts to 'pos',
        which will highlight only values higher than the expexted
        frequencies; 'all' will highlight both extremes (positive and
        negative); and an integer, which will use the first n entries,
        positive and negative, regardless of whether Z is higher than
        the conf_level or not.

    limit_N: sets a limit to N for the calculation of the Z score
        if the sample is too big. Defaults to None.

    MSE: calculates the Mean Square Error of the sample; defaults to
        False.

    show_plot: draws the test plot.

    '''
    if not isinstance(data, Analysis):
        data = Analysis(data, sign=sign, dec=dec, inform=inform)

    data = data.second_digit(inform=inform, MAD=MAD, conf_level=conf_level,
                             high_Z=high_Z, limit_N=limit_N, MSE=MSE,
                             show_plot=show_plot, ret_df=True)
    if inform:
        return data.sort_values('Z_test', ascending=False)
    else:
        return data


def last_two_digits(data, sign='all', dec=2, inform=True,
                    MAD=True, conf_level=95, high_Z='pos', limit_N=None,
                    MSE=False, show_plot=True):
    '''
    Performs the Last Two Digits test on the series of
    numbers provided.

    Parameters
    ----------

    data: sequence of numbers to be evaluated. Must be a numpy 1D array,
        a pandas Series or a pandas DataFrame column,with values being
        integers or floats.

    dec: number of decimal places to consider. Defaluts to 2.
        If integers, set to 0.

    inform: tells the number of registries that are being subjected to
        the Analysis and returns tha analysis DataFrame sorted by the
        highest Z score down. Defaults to True.

    MAD: calculates the Mean Absolute Difference between the
        found and the expected distributions; defaults to True.

    conf_level: confidence level to draw lower and upper limits when
        plotting and to limit the top deviations to show. Defaults to 95.
        If None, no boundaries will be drawn.

    high_Z: chooses which Z scores to be used when displaying results,
        according to the confidence level chosen. Defaluts to 'pos',
        which will highlight only values higher than the expexted
        frequencies; 'all' will highlight both extremes (positive and
        negative); and an integer, which will use the first n entries,
        positive and negative, regardless of whether Z is higher than
        the conf_level or not.

    limit_N: sets a limit to N for the calculation of the Z score
        if the sample is too big. Defaults to None.

    MSE: calculates the Mean Square Error of the sample; defaults to
        False.

    show_plot: draws the test plot.

    '''
    if not isinstance(data, Analysis):
        data = Analysis(data, sign=sign, dec=dec, inform=inform)

    data = data.last_two_digits(inform=inform, MAD=MAD, conf_level=conf_level,
                                high_Z=high_Z, limit_N=limit_N, MSE=MSE,
                                show_plot=show_plot, ret_df=True)
    if inform:
        return data.sort_values('Z_test', ascending=False)
    else:
        return data


def mantissas(data, inform=True, show_plot=True):
    '''
    Returns a Series with the data mantissas,

    Parameters
    ----------
    data: sequence to compute mantissas from, numpy 1D array, pandas
        Series of pandas DataFrame column.

    show_plot: plots the ordered mantissas and a line with the expected
        inclination. Defaults to True.
    '''
    mant = Mantissas(data)
    if inform:
        mant.inform()
    if show_plot:
        mant.show_plot()
    return mant


def summation(data, digs=2, sign='all', dec=2, top=20, inform=True,
              show_plot=True):
    '''
    Performs the Summation test. In a Benford series, the sums of the
    entries begining with the same digits tends to be the same.
    Works only with the First Digits (1, 2 or 3) test.

    Parameters
    ----------

    digs: tells the first digits to use: 1- first; 2- first two;
        3- first three. Defaults to 2.

    top: choses how many top values to show. Defaults to 20.

    show_plot: plots the results. Defaults to True.

    '''
    if not isinstance(data, Analysis):
        data = Analysis(data, sign=sign, dec=dec, inform=inform)

    data = data.summation(digs=digs, top=top, inform=inform,
                          show_plot=show_plot, ret_df=True)
    if inform:
        return data.sort_values('AbsDif', ascending=False)
    else:
        return data


def mad(data, test, sign='all', dec=2):
    '''
    '''
    if test not in [1, 2, 3, 22, -2]:
        raise ValueError('test was set to {0}. Should be 1, 2, 3, 22 or -2'.
                         format(test))
    start = Analysis(data, sign=sign, dec=dec, inform=False)
    if test in [1, 2, 3]:
        start.first_digits(digs=test, inform=False, MAD=True, simple=True)
    elif test == 22:
        start.second_digit(inform=False, MAD=True, simple=True)
    else:
        start.last_two_digits(inform=False, MAD=True, simple=True)
    return start.MAD


def mse(data, test, sign='all', dec=2):
    '''
    '''
    if test not in [1, 2, 3, 22, -2]:
        raise ValueError('test was set to {0}. Should be 1, 2, 3, 22 or -2'.
                         format(test))
    start = Analysis(data, sign=sign, dec=dec, inform=False)
    if test in [1, 2, 3]:
        start.first_digits(digs=test, MAD=False, MSE=True, simple=True)
    elif test == 22:
        start.second_digit(MAD=False, MSE=True, simple=True)
    else:
        start.last_two_digits(MAD=False, MSE=True, simple=True)
    return start.MSE


def duplicates():
    pass


def second_order():
    pass


def map_back():
    pass


def _inform_(df, high_Z, conf):
    '''
    Selects and sorts by the Z_stats chosen to be considered, informing or not,
    and populating the maps dict for further back analysis of the entries.
    '''

    if isinstance(high_Z, int):
        if conf is not None:
            dd = df[['Expected', 'Found', 'Z_test'
                     ]].sort_values('Z_test', ascending=False
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
            m2 = df.Z_test > conf
            dd = df[['Expected', 'Found', 'Z_test'
                     ]].loc[m1 & m2].sort_values('Z_test', ascending=False)
            print('\nThe entries with the significant positive deviations \
are:\n')
        elif high_Z == 'neg':
            m1 = df.Dif < 0
            m2 = df.Z_test > conf
            dd = df[['Expected', 'Found', 'Z_test'
                     ]].loc[m1 & m2].sort_values('Z_test', ascending=False)
            print('\nThe entries with the significant negative deviations \
are:\n')
        else:
            dd = df[['Expected', 'Found', 'Z_test'
                     ]].loc[df.Z_test > conf].sort_values('Z_test',
                                                          ascending=False)
            print('\nThe entries with the significant deviations are:\n')
    print(dd)

# to do:

# XXXXXXX MAD GENERAL FUNCTION XXXXXX

# XXXXXXX SECOND ORDER GENERAL FUNCTION XXXXXXX

# XXXXXXX MAPPING BACK XXXXXXX

# XXXXXX DUPLICATES XXXXXX