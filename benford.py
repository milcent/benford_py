'''
Benford_py for Python is a module for application of Benford's Law
to a sequence of numbers.

Dependent on pandas and numpy, using matplotlib for visualization

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

# Color pallete Island Jungle by Shy_violet
# http://www.colourlovers.com/palette/4348209/Island_Jungle

# Imports
from __future__ import print_function
from __future__ import division


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
            p = self.plot(kind='bar', color='#51702C',
                          figsize=(2 * (digs ** 2 + 5), 1.5 * (digs ** 2 + 5)))
            p.set_axis_bgcolor('#DDDFD2')


class Second(pd.DataFrame):
    '''
    Returns the expected probabilities of the Second Digits
    according to Benford's distribution.

    -> plot: option to plot a bar chart of the Expected proportions.
            Defaults to True.
    '''
    def __init__(self, plot=True):
        a = np.arange(10, 100)
        Expe = np.log10(1 + (1. / a))
        Sec_Dig = np.array(list(range(10)) * 9)

        pd.DataFrame.__init__(self, {'Expected': Expe, 'Sec_Dig': Sec_Dig})
        self = self.groupby('Sec_Dig').sum()
        if plot:
            p = self.plot(kind='bar', color='#51702C',
                          figsize=(14, 10.5), ylim=(0, .14))
            p.set_axis_bgcolor('#DDDFD2')


class LastTwo(pd.DataFrame):
    '''
    Returns the expected probabilities of the Last Two Digits
    according to Benford's distribution.

    -> plot: option to plot a bar chart of the Expected proportions.
            Defaults to True.
    '''
    def __init__(self, plot=True):
        exp = np.array([1 / 99.] * 100)
        pd.DataFrame.__init__(self, {'Expected': exp, 'Last_2_Dig': _lt_()})
        self.set_index('Last_2_Dig', inplace=True)
        if plot:
            p = self.plot(kind='bar', figsize=(15, 8), color='#51702C',
                          ylim=(0, 0.013))
            p.set_axis_bgcolor('#DDDFD2')


class Analysis(pd.DataFrame):
    '''
    Initiates the Analysis of the series. pandas DataFrame subclass.
    Values must be integers or floats. If not, it will try to convert
    them. If it does not succeed, a TypeError will be raised.
    A pandas DataFrame will be constructed, with the columns: original
    numbers without floating points, first, second, first two, first three
    and    last two digits, so the following tests will run properly.

    -> data: sequence of numbers to be evaluated. Must be in absolute values,
            since negative values with minus signs will distort the tests.
            XXXXXX---PONDERAR DEIXAR O COMANDO DE CONVERTER PARA AbS
    -> dec: number of decimal places to be accounted for. Especially important
            for the last two digits test. The numbers will be multiplied by
            10 to the power of the dec value. Defaluts to 2 (currency). If
            the numbers are integers, assign 0.
    -> sec_order: choice for the Second Order Test, which cumputes the
            differences between the ordered entries before running the Tests.
    -> inform: tells the number of registries that are being subjected to
            the Analysis; defaults to True
    -> latin: used for str dtypes representing numbers in latin format, with
            '.' for thousands and ',' for decimals. Converts to a string with
            only '.' for decimals if float, and none if int, so it can be later
            converted to a number format. Defaults to False.
    '''
    maps = {}  # dict for recording the indexes to be mapped back to the
    # dict of confidence levels for further use
    stats = {}
    confs = {'None': None, '80': 1.285, '85': 1.435, '90': 1.645, '95': 1.96,
             '99': 2.576, '99.9': 3.29, '99.99': 3.89, '99.999': 4.417,
             '99.9999': 4.892, '99.99999': 5.327}
    digs_dict = {'1': 'F1D', '2': 'F2D', '3': 'F3D'}

    def __init__(self, data, sign='all', dec=2, sec_order=False, inform=True,
                 latin=False):
        if sign not in ['all', 'pos', 'neg']:
            raise ValueError("The -sign- argument must be 'all','pos'\
 or 'neg'.")

        pd.DataFrame.__init__(self, {'Seq': data})
        # self.dropna(inplace=True)

        if self.Seq.dtypes != 'float' and self.Seq.dtypes != 'int':
            print('Sequence dtype is not int nor float.\nTrying \
to convert...\n')
            if latin:
                if dec != 0:
                    self.Seq = self.Seq.apply(_sanitize_latin_float_, dec=dec)
                else:
                    self.Seq = self.Seq.apply(_sanitize_latin_int_)
            # Try to convert to numbers
            self.Seq = self.Seq.convert_objects(convert_numeric=True)
            self.dropna(inplace=True)
            if self.Seq.dtypes == 'float' or self.Seq.dtypes == 'int':
                print('Conversion successful!')
            else:
                raise TypeError("The sequence dtype was not int nor float and\
 could not be converted.\nConvert it to whether int of float, or set latin to\
 True, and try again.")

        if sign == 'pos':
            self.Seq = self.Seq[self.Seq > 0]
        elif sign == 'neg':
            self.Seq = self.Seq[self.Seq < 0]
        else:
            self.Seq = self.Seq[self.Seq != 0]

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

        self['ZN'] = np.abs(self.Seq * (10**dec))  # dec - to manage decimals

        if dec != 0:
            self.ZN = self.ZN.apply(_tint_)

        # Extracts the digits in their respective positions,
        self['S'] = self.ZN.astype(str)
        self['F1D'] = self.S.str[:1]   # get the first digit
        self['SD'] = self.S.str[1:2]  # get the second digit
        self['F2D'] = self.S.str[:2]  # get the first two digits
        self['F3D'] = self.S.str[:3]  # get the first three digits
        self['L2D'] = self.S.str[-2:]  # get the last two digits
        # Leave the last two digits as strings , so as to be able to\
        # display '00', '01', ... up to '09', till '99'
        # convert the others to integers
        self.F1D = self.F1D.apply(_tint_)
        self.SD = self.SD.apply(_tint_)
        self.F2D = self.F2D.apply(_tint_)
        self.F3D = self.F3D.apply(_tint_)
        del self['S']
        # self = self[self.F2D>=10]

    def mantissas(self, plot=True, figsize=(15, 8)):
        '''
        Calculates the logs base 10 of the numbers in the sequence and Extracts
        the mantissae, which are the decimal parts of the logarithms. It them
        calculates the mean and variance of the mantissae, and compares them
        with the mean and variance of a Benford's sequence.
        plot -> plots the ordered mantissae and a line with the expected
                inclination. Defaults to True.
        figsize -> tuple that give sthe size of the figure when plotting
        '''
        self['Mant'] = _getMantissas_(self.Seq)
        p = self[['Seq', 'Mant']]
        p = p[p.Seq > 0].sort_values('Mant')
        print("The Mantissas MEAN is {0}. Ref: 0.5.".format(p.Mant.mean()))
        print("The Mantissas VARIANCE is {0}. Ref: 0.83333.".format(
              p.Mant.var()))
        N = len(p)
        # eturn p
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

    def second_digit(self, inform=True, MAD=True, conf_level=95,
                     MSE=False, show_high_Z='pos', limit_N=None, plot=True):
        '''
        Performs the Benford Second Digit test with the series of
        numbers provided.

        inform -> tells the number of registries that are being subjected to
            the Analysis; defaults to True

        MAD -> calculates the Mean of the Absolute Differences between the
        found and the expected distributions; defaults to True.

        conf_level -> confidence level to draw lower and upper limits when
            plotting and to limit the mapping of the proportions to only the
            ones significantly diverging from the expected. Defaults to 95.
            If None, no boundaries will be drawn.

        show_high_Z -> chooses which Z scores to be used when displaying
            results, according to the confidence level chosen. Defaluts to
            'pos', which will highlight only the values that are higher than
            the expexted frequencies; 'all' will highlight both found
            extremes (positive and negative); and an integer, which will use
            the first n entries, positive and negative, regardless of whether
            Z is higher than the conf_level Z or not

        limit_N -> sets a limit to N for the calculation of the Z statistic,
            which suffers from the power problem when the sampl is too large.
            Usually, the N is set to a maximum 2,500. Defaults to None.

        MSE -> calculate the Mean Square Error of the sample; defaluts to
            False.

        plot -> draws the plot of test for visual comparison, with the found
            distributions in bars and the expected ones in a line.

        '''
        if str(conf_level) not in list(self.confs.keys()):
            raise ValueError("Value of -conf_level- must be one of the\
 following: {0}".format(list(self.confs.keys())))

        temp = self.loc[self.ZN >= 10]

        # Assigning to N the superior limit or the lenght of the series
        if limit_N is None or limit_N > len(temp):
            N = len(temp)
        # Check on limit_N being a positive integer
        else:
            if limit_N < 0 or not isinstance(limit_N, int):
                raise ValueError("-limit_N- must be None or a positive\
 integer.")
            else:
                N = limit_N

        conf = self.confs[str(conf_level)]

        x = np.arange(0, 10)

        if inform:
            print("\nTest performed on {0} registries.\nDiscarded \
{1} records < 10 after preparation.".format(N, N - len(temp)))
        # get the number of occurrences of each second digit
        v = temp.SD.value_counts()
        # get their relative frequencies
        p = temp.SD.value_counts(normalize=True)
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
        df['Z_test'] = _Z_test(df, N)

        # Populate dict with the most relevant entries
        self.maps['SD'] = np.array(_inform_and_map_(df, inform,
                                   show_high_Z, conf))

        # Mean absolute difference
        if MAD:
            self.stats['SD_MAD'] = _mad_(df, 'SD', inform=inform)

        # Mean Square Error
        if MSE:
            self.stats['SD_MSE'] = _mse_(df, inform=inform)

        # Plotting the expected frequncies (line) against the found ones(bars)
        if plot:
            _plot_dig_(df, x=x, y_Exp=df.Expected, y_Found=df.Found,
                       N=N, figsize=(10, 6), conf_Z=conf)

        # return df

    def first_digits(self, digs, inform=True, MAD=True, conf_level=95,
                     show_high_Z='pos', limit_N=None, MSE=False, plot=True):
        '''
        Performs the Benford First Digits test with the series of
        numbers provided, and populates the mapping dict for future
        selection of the original series.

        digs -> number of first digits to consider. Must be 1 (first digit),
            2 (first two digits) or 3 (first three digits).

        inform -> tells the number of registries that are being subjected to
            the Analysis; defaults to True

        MAD -> calculates the Mean of the Absolute Differences between the
            found and the expected distributions; defaults to True.

        conf_level -> confidence level to draw lower and upper limits when
            plotting and to limit the mapping of the proportions to only the
            ones significantly diverging from the expected. Defaults to 95.
            If None, no boundaries will be drawn.

        show_high_Z -> chooses which Z scores to be used when displaying
            results, according to the confidence level chosen. Defaluts to
            'pos', which will highlight only the values that are higher than
            the expexted frequencies; 'all' will highlight both found
            extremes (positive and negative); and an integer, which will use
            the first n entries, positive and negative, regardless of whether
            Z is higher than the conf_level Z or not.

        limit_N -> sets a limit to N for the calculation of the Z statistic,
            which suffers from the power problem when the sampl is too large.
            Usually, N is set to a maximum 2,500. Defaults to None.

        MSE -> calculates the Mean Square Error of the sample; defaults to
            False.

        plot -> draws the test plot for visual comparison, with the found
            distributions in bars and the expected ones in a line.


        '''
        # Check on the possible values for confidence lavels
        if str(conf_level) not in list(self.confs.keys()):
            raise ValueError("Value of parameter -conf_level- must be one\
 of the following: {0}".format(list(self.confs.keys())))
        # Check on possible digits
        if digs not in [1, 2, 3]:
            raise ValueError("The value assigned to the parameter -digs-\
 was {0}. Value must be 1, 2 or 3.".format(digs))

        temp = self.loc[self.ZN >= 10 ** (digs - 1)]

        # Assigning to N the superior limit or the lenght of the series
        if limit_N is None or limit_N > len(temp):
            N = len(temp)
        # Check on limit_N being a positive integer
        else:
            if limit_N < 0 or not isinstance(limit_N, int):
                raise ValueError("-limit_N- must be None or a positive\
 integer.")
            else:
                N = limit_N

        dig_name = 'F{0}D'.format(digs)
        n, m = 10 ** (digs - 1), 10 ** (digs)
        x = np.arange(n, m)
        conf = self.confs[str(conf_level)]

        if inform:
            print("\nTest performed on {0} registries.\nDiscarded {1} \
records < {2} after preparation.".format(len(self), len(self) - len(temp),
                                         10 ** (digs - 1)))
        # get the number of occurrences of the first two digits
        v = temp[dig_name].value_counts()
        # get their relative frequencies
        p = temp[dig_name].value_counts(normalize=True)
        # crate dataframe from them
        df = pd.DataFrame({'Counts': v, 'Found': p}).sort_index()
        # reindex from n to m in the case one or more of the first
        # digits are missing, so the Expected frequencies column
        # can later be joined; and swap NANs with zeros.
        if len(df.index) < m - n:
            df = df.reindex(x).fillna(0)
        # join the dataframe with the one of expected Benford's frequencies
        df = First(digs=digs, plot=False).join(df)
        # create column with absolute differences
        df['Dif'] = df.Found - df.Expected
        df['AbsDif'] = np.absolute(df.Dif)
        # calculate the Z-test column an display the dataframe by descending
        # Z test
        df['Z_test'] = _Z_test(df, N)
        # Populate dict with the most relevant entries
        self.maps[dig_name] = np.array(_inform_and_map_(df, inform,
                                       show_high_Z, conf))

        # Mean absolute difference
        if MAD:
            self.stats['{0}_MAD'.format(dig_name)] = _mad_(df, test=dig_name,
                                                           inform=inform)

        # Mean Square Error
        if MSE:
            self.stats['{0}_MSE'.format(dig_name)] = _mse_(df, inform=inform)

        # Plotting the expected frequncies (line) against the found ones(bars)
        if plot:
            _plot_dig_(df, x=x, y_Exp=df.Expected, y_Found=df.Found, N=N,
                       figsize=(2 * (digs ** 2 + 5), 1.5 * (digs ** 2 + 5)),
                       conf_Z=conf)

        # return df

    def last_two_digits(self, inform=True, MAD=False, conf_level=95,
                        show_high_Z='pos', limit_N=None, MSE=False, plot=True):
        '''
        Performs the Benford Last Two Digits test with the series of
        numbers provided.

        inform -> tells the number of registries that are being subjected to
            the Analysis; defaults to True

        MAD -> calculates the Mean of the Absolute Differences between the
            found and the expected distributions; defaults to False.

        conf_level -> confidence level to draw lower and upper limits when
            plotting and to limit the mapping of the proportions to only the
            ones significantly diverging from the expected. Defaults to 95.
            If None, no boundaries will be drawn.

        show_high_Z -> chooses which Z scores to be used when displaying
            results, according to the confidence level chosen. Defaluts to
            'pos', which will highlight only the values that are higher than
            the expexted frequencies; 'all' will highlight both found extremes
            (positive and negative); and an integer, which will use the first
            n entries, positive and negative,regardless of whether the Z is
            higher than the conf_level Z or not

        limit_N -> sets a limit to N for the calculation of the Z statistic,
            which suffers from the power problem when the sampl is too large.
            Usually, the Nis set to a maximum 2,500. Defaults to None.

        MSE -> calculates the Mean Square Error of the sample; defaluts to
            False.

        plot -> draws the test plot for visual comparison, with the found
            distributions in bars and the expected ones in a line.

        '''
        if str(conf_level) not in list(self.confs.keys()):
            raise ValueError("Value of -conf_level- must be one of the \
following: {0}".format(list(self.confs.keys())))

        temp = self.loc[self.ZN >= 1000]

        # Assigning to N the superior limit or the lenght of the series
        if limit_N is None or limit_N > len(temp):
            N = len(temp)
        # Check on limit_N being a positive integer
        else:
            if limit_N < 0 or not isinstance(limit_N, int):
                raise ValueError("-limit_N- must be None or a positive \
integer.")
            else:
                N = limit_N

        conf = self.confs[str(conf_level)]

        x = np.arange(0, 100)
        if inform:
            print("\nTest performed on {0} registries.\nDiscarded {1} \
records < 1000 after preparation".format(len(self), len(self) - len(temp)))
        # get the number of occurrences of the last two digits
        v = temp.L2D.value_counts()
        # get their relative frequencies
        p = temp.L2D.value_counts(normalize=True)
        # crate dataframe from them
        df = pd.DataFrame({'Counts': v, 'Found': p}).sort_index()
        # join the dataframe with the one of expected Benford's frequencies
        df = LastTwo(plot=False).join(df)
        # create column with absolute differences
        df['Dif'] = df.Found - df.Expected
        df['AbsDif'] = np.absolute(df.Dif)
        # calculate the Z-test column an display the dataframe by descending
        # Z test
        df['Z_test'] = _Z_test(df, N)

        # Populate dict with the most relevant entries
        self.maps['L2D'] = np.array(_inform_and_map_(df, inform,
                                    show_high_Z, conf)).astype(int)

        # Mean absolute difference
        if MAD:
            self.stats['L2D_MAD'] = _mad_(df, test='L2D', inform=inform)

        # Mean Square Error
        if MSE:
            self.stats['L2D_MSE'] = _mse_(df, inform=inform)

        # Plotting expected frequencies (line) versus found ones (bars)
        if plot:
            _plot_dig_(df, x=x, y_Exp=df.Expected, y_Found=df.Found, N=N,
                       figsize=(15, 8), conf_Z=conf, text_x=True)

        # return df

    def summation(self, digs=2, top=20, inform=True, plot=True):
        '''
        Performs the Summation test. In a Benford series, the sums of the
        entries begining with the same digits tends to be the same.

        digs -> tells the first digits to use. 1- first; 2- first two;
                3- first three. Defaults to 2.

        top -> choses how many top values to show. Defaults to 20.

        plot -> plots the results. Defaults to True.
        '''

        if digs not in [1, 2, 3]:
            raise ValueError("The value assigned to the parameter -digs-\
 was {0}. Value must be 1, 2 or 3.".format(digs))
        # Set the future dict key
        if inform:
            # N = len(self)
            print("\nTest performed on {0} registries.\n".format(len(self)))
        dig_name = 'SUM{0}'.format(digs)
        if digs == 1:
            top = 9
        # Call the dict for F1D, F2D, F3D
        d = self.digs_dict[str(digs)]
        # Call the expected proportion according to digs
        li = 1. / (9 * (10 ** (digs - 1)))

        s = self.groupby(d).sum()
        # s.drop(0, inplace=True)
        s['Percent'] = s.ZN / s.ZN.sum()
        s.columns.values[1] = 'Sum'
        s = s[['Sum', 'Percent']]
        s['AbsDif'] = np.absolute(s.Percent - li)

        # Populate dict with the most relevant entries
        self.maps[dig_name] = np.array(_inform_and_map_(s, inform,
                                       show_high_Z=top, conf=None)).astype(int)

        if plot:
            # f = {'1': (8, 5), '2': (13, 8), '3': (21, 13)}
            _plot_sum_(s, figsize=(
                       2 * (digs ** 2 + 5), 1.5 * (digs ** 2 + 5)), li=li)

        # return

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
    Returns the Mean Absolute Deviation (MAD) of the found proportions from the
    expected proportions. Then, it compares the found MAD with the accepted
    ranges of the respective test.

    frame -> DataFrame with the Absolute Deviations already calculated.
    test -> Teste applied (F1D, SD, F2D...)
    inform -> prints the MAD result and compares to limit values of
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
    log_a = np.log10(arr)
    return np.abs(log_a) - log_a.astype(int)  # the number - its integer part


def _lt_():
    li = []
    d = '0123456789'
    for i in d:
        for j in d:
            t = i + j
            li.append(t)
    return np.array(li)


def _to_float_(st):
    try:
        return float(st) / 100
    except:
        return np.nan


def _sanitize_(arr):
    '''
    Prepares the series to enter the test functions, in case pandas
    has not inferred the type to be float, especially when parsing
    from latin datases which use '.' for thousands and ',' for the
    floating point.
    '''
    return pd.Series(arr).dropna().apply(str).apply(
        _only_numerics_).apply(_l_0_strip_)


def _only_numerics_(seq):
    '''
    From a str sequence, return the characters that represent numbers

    seq -> string sequence
    '''
    return list(filter(type(seq).isdigit, seq))


def _str_to_float_(s):
    if '.' in s or ',' in s:
        s = list(filter(type(s).isdigit, s))
        s = s[:-2] + '.' + s[-2:]
        return float(s)
    else:
        if list(filter(type(s).isdigit, s)) == '':
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
    bars = plt.bar(x, y_Found * 100., color='#3D959F', label='Found', zorder=3)
    ax.set_xticks(x + .4)
    ax.set_xticklabels(x)
    ax.plot(x, y_Exp * 100., color='#284324', linewidth=2.5,
            label='Benford', zorder=4)
    # ax.grid(axis='y', color='w', linestyle='-', zorder=0)
    ax.set_axis_bgcolor('#DDDFD2')
    ax.legend()
    if text_x:
        plt.xticks(x, df.index, rotation='vertical')
    # Plotting the Upper and Lower bounds considering the Z for the
    # informed confidence level
    if conf_Z is not None:
        sig = conf_Z * np.sqrt(y_Exp * (1 - y_Exp) / N)
        upper = y_Exp + sig + (1 / (2 * N))
        lower = y_Exp - sig - (1 / (2 * N))
        u = (y_Found < lower) | (y_Found > upper)
        for i, b in enumerate(bars):
            if u.iloc[i]:
                b.set_color('#990007')
        lower *= 100.
        upper *= 100.
        ax.plot(x, upper, color='#284324', zorder=5)
        ax.plot(x, lower, color='#284324', zorder=5)
        ax.fill_between(x, upper, lower, color='#284324', alpha=.3)
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
    ax.bar(df.index, df.Percent, color='#3D959F', label='Found Sums', zorder=3)
    ax.axhline(li, color='#284324', linewidth=2, label='Expected', zorder=4)
    ax.set_axis_bgcolor('#DDDFD2')
    # ax.grid(axis='y', color='w', linestyle='-', zorder=0)
    ax.legend()


def _sanitize_latin_float_(s, dec=2):
    s = str(s)
    s = list(filter(type(s).isdigit, s))
    return s[:-dec] + '.' + s[-dec:]


def _sanitize_latin_int_(s):
    s = str(s)
    s = list(filter(type(s).isdigit, s))
    return s


def _inform_and_map_(df, inform, show_high_Z, conf):
    '''
    Selects and sorts by the Z_stats chosen to be considered, informing or not,
    and populating the maps dict for further back analysis of the entries.
    '''

    if inform:
        if isinstance(show_high_Z, int):
            if conf is not None:
                dd = df[['Expected', 'Found', 'Z_test'
                         ]].sort_values('Z_test', ascending=False
                                        ).head(show_high_Z)
                print('\nThe entries with the top {0} Z scores are\
:\n'.format(show_high_Z))
            # Summation Test
            else:
                dd = df.sort_values('AbsDif', ascending=False
                                    ).head(show_high_Z)
                print('\nThe entries with the top {0} absolute deviations\
 are:\n'.format(show_high_Z))
        else:
            if show_high_Z == 'pos':
                m1 = df.Dif > 0
                m2 = df.Z_test > conf
                dd = df[['Expected', 'Found', 'Z_test'
                         ]][m1 & m2].sort_values('Z_test', ascending=False)
                print('\nThe entries with the significant positive deviations\
 are:\n')
            elif show_high_Z == 'neg':
                m1 = df.Dif < 0
                m2 = df.Z_test > conf
                dd = df[['Expected', 'Found', 'Z_test'
                         ]][m1 & m2].sort_values('Z_test', ascending=False)
                print('\nThe entries with the significant negative deviations\
 are:\n')
            else:
                dd = df[['Expected', 'Found', 'Z_test'
                         ]][df.Z_test > conf].sort_values('Z_test',
                                                          ascending=False)
                print('\nThe entries with the significant deviations are:\n')
        print(dd)
        return dd.index
    else:
        if isinstance(show_high_Z, int):
            if conf is not None:
                dd = df[['Expected', 'Found', 'Z_test'
                         ]].sort_values('Z_test', ascending=False
                                        ).head(show_high_Z)
            # Summation Test
            else:
                dd = df.sort_values('AbsDif', ascending=False
                                    ).head(show_high_Z)
        else:
            if show_high_Z == 'pos':
                dd = df[['Expected', 'Found', 'Z_test'
                         ]][df.Dif > 0 and df.Z_test > conf
                            ].sort_values('Z_test', ascending=False)
            elif show_high_Z == 'neg':
                dd = df[['Expected', 'Found', 'Z_test'
                         ]][df.Dif < 0 and df.Z_test > conf
                            ].sort_values('Z_test', ascending=False)
            else:
                dd = df[['Expected', 'Found', 'Z_test'
                         ]][df.Z_test > conf
                            ].sort_values('Z_test', ascending=False)
        return dd.index
