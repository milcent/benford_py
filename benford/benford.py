import warnings
from pandas import Series, DataFrame
from numpy import arange, log10, ones, abs, cos, sin, pi, mean
from .constants import confs, digs_dict, sec_order_dict, rev_digs, names, \
    mad_dict, crit_chi2, KS_crit
from .checks import _check_digs_, _check_confidence_, _check_test_, \
    _check_num_array_, _check_high_Z_
from .utils import _set_N_, input_data, prepare, \
    subtract_sorted, prep_to_roll, mad_to_roll, mse_to_roll, \
    get_mantissas
from .expected import First, Second, LastTwo, _test_
from .viz import _get_plot_args, plot_digs, plot_sum, plot_ordered_mantissas,\
    plot_mantissa_arc_test, plot_roll_mse, plot_roll_mad
from .reports import _inform_, _report_mad_, _report_test_, _deprecate_inform_,\
    _report_mantissa_
from .stats import Z_score, chi_sq, chi_sq_2, kolmogorov_smirnov,\
    kolmogorov_smirnov_2


class Base(DataFrame):
    """Internalizes and prepares the data for Analysis.

    Args:
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

    Raises:
        TypeError: if not receiving `int` or `float` as input.
    """

    def __init__(self, data, decimals, sign='all', sec_order=False):

        DataFrame.__init__(self, {'seq': data})

        if (self.seq.dtypes != 'float64') & (self.seq.dtypes != 'int64'):
            raise TypeError("The sequence dtype was not pandas int64 nor "
                            "float64. Convert it to whether int of float, "
                            "and try again.")

        if sign == 'all':
            self.seq = self.seq.loc[self.seq != 0]
        elif sign == 'pos':
            self.seq = self.seq.loc[self.seq > 0]
        else:
            self.seq = self.seq.loc[self.seq < 0]

        self.dropna(inplace=True)

        ab = self.seq.abs()

        if self.seq.dtypes == 'int64':
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
            self[col] = (temp // 10 ** ((log10(temp).astype(int)) -
                                        (rev_digs[col] - 1)))
            # fill NANs with -1, which is a non-usable value for digits,
            # to be discarded later.
            self[col] = self[col].fillna(-1).astype(int)
        # Second digit
        temp_sd = self.loc[self.ZN >= 10]
        self['SD'] = (temp_sd.ZN // 10**((log10(temp_sd.ZN)).astype(int) -
                                         1)) % 10
        self['SD'] = self['SD'].fillna(-1).astype(int)
        # Last two digits
        temp_l2d = self.loc[self.ZN >= 1000]
        self['L2D'] = temp_l2d.ZN % 100
        self['L2D'] = self['L2D'].fillna(-1).astype(int)


class Test(DataFrame):
    """Transforms the original number sequence into a DataFrame reduced
    by the ocurrences of the chosen digits, creating other computed
    columns

    Args:
        base: The Base object with the data prepared for Analysis
        digs: Tells which test to perform: 1: first digit; 2: first two digits;
            3: furst three digits; 22: second digit; -2: last two digits.
        confidence: confidence level to draw lower and upper limits when
            plotting and to limit the top deviations to show.
        limit_N: sets a limit to N as the sample size for the calculation of
            the Z scores if the sample is too big. Defaults to None.

    Attributes:
        N: Number of records in the sample to consider in computations
        ddf: Degrees of Freedom to look up for the critical chi-square value
        chi_square: Chi-square statistic for the given test
        KS: Kolmogorov-Smirnov statistic for the given test
        MAD: Mean Absolute Deviation for the given test
        confidence: Confidence level to consider when setting some critical values
        digs (int): numerical representation of the test at hand. 1: F1D; 2: F2D;
            3: F3D; 22: SD; -2: L2D.
        sec_order (bool): True if the test is a Second Order one
    """

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
        self['AbsDif'] = self.Dif.abs()
        self.N = _set_N_(len(base), limit_N)
        self['Z_score'] = Z_score(self, self.N)
        self.ddf = len(self) - 1
        self.chi_square = chi_sq_2(self)
        self.KS = kolmogorov_smirnov_2(self)
        self.MAD = self.AbsDif.mean()
        self.MSE = (self.AbsDif ** 2).mean()
        self.confidence = confidence
        self.digs = digs
        self.sec_order = sec_order

        if sec_order:
            self.name = names[sec_order_dict[digs]]
        else:
            self.name = names[digs_dict[digs]]

    def update_confidence(self, new_conf, check=True):
        """Sets a new confidence level for the Benford object, so as to be used to
        produce critical values for the tests

        Args:
            new_conf: new confidence level to draw lower and upper limits when
                plotting and to limit the top deviations to show, as well as to
                calculate critical values for the tests' statistics.
            check: checks the value provided for the confidence. Defaults to True
        """
        if check:
            self.confidence = _check_confidence_(new_conf)
        else:
            self.confidence = new_conf

    @property
    def critical_values(self):
        """dict: a dictionary with the critical values for the test at hand,
            according to the current confidence level."""
        return {'Z': confs[self.confidence],
                'KS': KS_crit[self.confidence] / (self.N ** 0.5),
                'chi2': crit_chi2[self.ddf][self.confidence],
                'MAD': mad_dict[self.digs]
                }

    def show_plot(self, save_plot=None, save_plot_kwargs=None):
        """Draws the test plot.
        
        Args:
            save_plot: string with the path/name of the file in which the generated
                plot will be saved. Uses matplotlib.pyplot.savefig(). File format
                is infered by the file name extension.
            save_plot_kwargs: dict with any of the kwargs accepted by
                matplotlib.pyplot.savefig()
                https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
                Only available when save_plot is a string with the figure file
                path/name.
        """
        x, figsize, text_x = _get_plot_args(self.digs)
        plot_digs(self, x=x, y_Exp=self.Expected, y_Found=self.Found,
                    N=self.N, figsize=figsize, conf_Z=confs[self.confidence],
                    text_x=text_x, save_plot=save_plot, save_plot_kwargs=save_plot_kwargs
                    )

    def report(self, high_Z='pos', show_plot=True,
               save_plot=None, save_plot_kwargs=None):
        """Handles the report especific to the test, considering its statistics
        and according to the current confidence level.

        Args:
            high_Z: chooses which Z scores to be used when displaying results,
                according to the confidence level chosen. Defaluts to 'pos',
                which will highlight only values higher than the expexted
                frequencies; 'all' will highlight both extremes (positive and
                negative); and an integer, which will use the first n entries,
                positive and negative, regardless of whether Z is higher than
                the critical value or not.
            show_plot: calls the show_plot method, to draw the test plot
            save_plot: string with the path/name of the file in which the generated
                plot will be saved. Uses matplotlib.pyplot.savefig(). File format
                is infered by the file name extension. Only available when
                plot=True.
            save_plot_kwargs: dict with any of the kwargs accepted by
                matplotlib.pyplot.savefig()
                https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
                Only available when plot=True and save_plot is a string with the
                figure file path/name.
        """
        high_Z = _check_high_Z_(high_Z)
        _report_test_(self, high_Z, self.critical_values)
        if show_plot:
            self.show_plot(save_plot=save_plot, save_plot_kwargs=save_plot_kwargs)


class Summ(DataFrame):
    """Gets the base object and outputs a Summation test object

    Args:
       base: The Base object with the data prepared for Analysis
       test: The test for which to compute the summation
    """

    def __init__(self, base, test):
        super(Summ, self).__init__(base.abs()
                                   .groupby(test)[['seq']]
                                   .sum())
        self['Percent'] = self.seq / self.seq.sum()
        self.columns.values[0] = 'Sum'
        self.expected = 1 / len(self)
        self['AbsDif'] = (self.Percent - self.expected).abs()
        self.index = self.index.astype(int)
        #: Mean Absolute Deviation for the test
        self.MAD = self.AbsDif.mean()
        self.MSE = (self.AbsDif ** 2).mean()
        #: Confidence level to consider when setting some critical values
        self.confidence = None
        # (int): numerical representation of the test at hand
        self.digs = rev_digs[test]
        # (str): the name of the Summation test.
        self.name = names[f'{test}_Summ']

    def show_plot(self, save_plot=None, save_plot_kwargs=None):
        """Draws the Summation test plot
        
        Args:
            save_plot: string with the path/name of the file in which the generated
                plot will be saved. Uses matplotlib.pyplot.savefig(). File format
                is infered by the file name extension.
            save_plot_kwargs: dict with any of the kwargs accepted by
                matplotlib.pyplot.savefig()
                https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
                Only available when save_plot is a string with the figure file
                path/name.
        """
        figsize=(2 * (self.digs ** 2 + 5), 1.5 * (self.digs ** 2 + 5))
        plot_sum(self, figsize, self.expected,
                 save_plot=save_plot, save_plot_kwargs=save_plot_kwargs)
    
    def report(self, high_diff=None, show_plot=True,
               save_plot=None, save_plot_kwargs=None):
        """Gives the report on the Summation test.

        Args:
            high_diff: Number of records to show after ordering by the absolute
                differences between the found and the expected proportions
            show_plot: calls the show_plot method, to draw the Summation test plot
            save_plot: string with the path/name of the file in which the generated
                plot will be saved. Uses matplotlib.pyplot.savefig(). File format
                is infered by the file name extension. Only available when
                plot=True.
            save_plot_kwargs: dict with any of the kwargs accepted by
                matplotlib.pyplot.savefig()
                https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
                Only available when plot=True and save_plot is a string with the
                figure file path/name.
        """
        _report_test_(self, high_diff)
        if show_plot:
            self.show_plot(save_plot=save_plot, save_plot_kwargs=save_plot_kwargs)


class Mantissas(object):
    """Computes and holds the mantissas of the logarithms of the records

    Args:
        data: sequence to compute mantissas from. numpy 1D array, pandas
            Series of pandas DataFrame column.
    """

    def __init__(self, data):

        data = Series(_check_num_array_(data))
        data = data.dropna().loc[data != 0].abs()
        #: (DataFrame): pandas DataFrame with the mantissas
        self.data = DataFrame({'Mantissa': get_mantissas(data.abs())})
        # (dict): Dictionary with the mantissas statistics
        self.stats = {'Mean': self.data.Mantissa.mean(),
                      'Var': self.data.Mantissa.var(),
                      'Skew': self.data.Mantissa.skew(),
                      'Kurt': self.data.Mantissa.kurt()}

    def report(self, show_plot=True, save_plot=None, save_plot_kwargs=None):
        """Displays the Mantissas test stats.

        Args:
            show_plot: shows the Ordered Mantissas plot and the Arc Test plot.
                Defaults to True.
            save_plot: string with the path/name of the file in which the generated
                plot will be saved. Uses matplotlib.pyplot.savefig(). File format
                is infered by the file name extension. Only available when
                plot=True.
            save_plot_kwargs: dict with any of the kwargs accepted by
                matplotlib.pyplot.savefig()
                https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
                Only available when plot=True and save_plot is a string with the
                figure file path/name.
        """
        _report_mantissa_(self.stats)

        if show_plot:
            self.show_plot(save_plot=save_plot, save_plot_kwargs=save_plot_kwargs)
            self.arc_test(save_plot=save_plot, save_plot_kwargs=save_plot_kwargs)

    def show_plot(self, figsize=(12, 6), save_plot=None, save_plot_kwargs=None):
        """Plots the ordered mantissas and a line with the expected
        inclination.

        Args:
            figsize (tuple): figure size dimensions
            save_plot: string with the path/name of the file in which the generated
                plot will be saved. Uses matplotlib.pyplot.savefig(). File format
                is infered by the file name extension.
            save_plot_kwargs: dict with any of the kwargs accepted by
                matplotlib.pyplot.savefig()
                https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
                Only available when save_plot is a string with the figure file
                path/name.
        """
        plot_ordered_mantissas(self.data.Mantissa, figsize=figsize,
                               save_plot=save_plot, save_plot_kwargs=save_plot_kwargs)

    def arc_test(self, decimals=2, grid=True, figsize=12,
                 save_plot=None, save_plot_kwargs=None):
        """Adds two columns to Mantissas's DataFrame equal to their "X" and "Y"
        coordinates, plots its to a scatter plot and calculates the gravity
        center of the circle.

        Args:
            decimals: number of decimal places for displaying the gravity center.
                Defaults to 2.
            grid: show grid of the plot. Defaluts to True.
            figsize (int): size of the figure to be displayed. Since it is a square,
                there is no need to provide a tuple, like is usually the case with
                matplotlib.
            save_plot: string with the path/name of the file in which the generated
                plot will be saved. Uses matplotlib.pyplot.savefig(). File format
                is infered by the file name extension. Only available when
                plot=True.
            save_plot_kwargs: dict with any of the kwargs accepted by
                matplotlib.pyplot.savefig()
                https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
                Only available when plot=True and save_plot is a string with the
                figure file path/name.
        """
        if self.stats.get('gravity_center') is None:
            self.data['mant_x'] = cos(2 * pi * self.data.Mantissa)
            self.data['mant_y'] = sin(2 * pi * self.data.Mantissa)
            self.stats['gravity_center'] = (self.data.mant_x.mean(),
                                            self.data.mant_y.mean())
        
        plot_mantissa_arc_test(self.data, self.stats, decimals=decimals, 
                               grid=grid, figsize=figsize,
                               save_plot=save_plot, save_plot_kwargs=save_plot_kwargs)


class Benford(object):
    """Initializes a Benford Analysis object and computes the proportions for
    the digits. The tets dataFrames are atributes, i.e., obj.F1D is the First
    Digit DataFrame, the obj.F2D,the First Two Digits one, and so one, F3D for
    First Three Digits, SD for Second  Digit and L2D for Last Two Digits.

    Args:
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

    Attributes:
        data: the raw data provided for the analysis
        chosen: the column of the DataFrame to be analysed or the data itself
        sign (str): which number sign(s) to include in the analysis
        confidence: current confidence level
        limit_N (int): sample size to use in computations
        verbose (bool): verbose or not
        base: the Base, pre-processed object
        tests (:obj:`list` of :obj:`str`): keeps track of the tests the
            instance has
    """

    def __init__(self, data, decimals=2, sign='all', confidence=95,
                 mantissas=False, sec_order=False, summation=False,
                 limit_N=None, verbose=True):
        self.data, self.chosen = input_data(data)
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
            print('\n', ' Benford Object Instantiated '.center(50, '#'), '\n')
            print(f'Initial sample size: {len(self.chosen)}.\n')
            print(f'Test performed on {len(self.base)} registries.\n')
            print(
                f'Number of discarded entries for each test:\n{self._discarded}')

        if mantissas:
            self.mantissas()

        if sec_order:
            self.sec_order()

        if summation:
            self.summation()

    def update_confidence(self, new_conf, tests=None):
        """Sets (a) new confidence level(s) for the Benford object, so as to be
        used to produce critical values for the tests.

        Args:
            new_conf: new confidence level to draw lower and upper limits when
                plotting and to limit the top deviations to show, as well as to
                calculate critical values for the tests' statistics.
            tests (:obj:`list` of :obj:`str`): list of tests names (strings) to
                have their confidence updated. If only one, provide a one-element
                list, like ['F1D']. Defauts to None, in which case it will use
                the instance .test list attribute.

        Raises:
            ValueError: if the test argument is not a `list` or `None`.
        """
        self.confidence = _check_confidence_(new_conf)
        if tests is None:
            tests = self.tests
        else:
            if not isinstance(tests, list):
                raise ValueError('tests must be a list or None.')
        for test in tests:
            try:
                getattr(self, test).update_confidence(
                    self.confidence, check=False)
            except AttributeError:
                if test in ['Mantissas', 'F1D_Summ', 'F2D_Summ', 'F3D_Summ']:
                    pass
                else:
                    print(
                        f"{test} not in Benford instance tests - review test's name.")
                    pass

    @property
    def all_confidences(self):
        """dict: a dictionary with a confidence level for each computed tests,
        when applicable."""
        con_dic = {}
        for key in self.tests:
            try:
                con_dic[key] = getattr(self, key).confidence
            except AttributeError:
                pass
        return con_dic

    def mantissas(self):
        """Adds a Mantissas object to the tests, with all its statistics and
        plotting capabilities.
        """
        self.Mantissas = Mantissas(self.base.seq)
        self.tests.append('Mantissas')
        if self.verbose:
            print('\nAdded Mantissas test.')

    def sec_order(self):
        """Runs the Second Order tests, which are the Benford's tests
        performed on the differences between the ordered sample (a value minus
        the one before it, and so on). If the original series is Benford-
        compliant, this new sequence should aldo follow Beford. The Second
        Order can also be called separately, through the method sec_order().
        """
        #: Base instance of the differences between the ordered sample
        self.base_sec = Base(subtract_sorted(self.chosen),
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
        """Creates Summation test DataFrames from Base object"""
        for test in ['F1D', 'F2D', 'F3D']:
            t = f'{test}_Summ'
            setattr(self, t, Summ(self.base, test))
            self.tests.append(t)

        if self.verbose:
            print('\nAdded Summation DataFrames to F1D, F2D and F3D Tests.')


class Source(DataFrame):
    """Prepares the data for Analysis. pandas DataFrame subclass.

    Args:
        data: sequence of numbers to be evaluated. Must be a numpy 1D array,
            a pandas Series or a pandas DataFrame column, with values being
            integers or floats.
        decimals: number of decimal places to consider. Defaluts to 2.
            If integers, set to 0. If set to -infer-, it will remove the zeros
            and consider up to the fifth decimal place to the right, but will
            loose performance.
        sign: tells which portion of the data to consider. pos: only the positive
            entries; neg: only negative entries; all: all entries but zeros.
            Defaults to all.
        sec_order: choice for the Second Order Test, which cumputes the
            differences between the ordered entries before running the Tests.
        verbose: tells the number of registries that are being subjected to
            the analysis; defaults to True.

    Raises:
        ValueError: if the `sign` arg is not in ['all', 'pos', 'neg']
        TypeError: if not receiving `int` or `float` as input.
    """

    def __init__(self, data, decimals=2, sign='all', sec_order=False,
                 verbose=True, inform=None):

        if sign not in ['all', 'pos', 'neg']:
            raise ValueError("The -sign- argument must be "
                             "'all','pos' or 'neg'.")

        DataFrame.__init__(self, {'seq': data})

        if self.seq.dtypes != 'float64' and self.seq.dtypes != 'int64':
            raise TypeError('The sequence dtype was not pandas int64 nor float64.\n'
                            'Convert it to whether int64 of float64, and try again.')

        if sign == 'pos':
            self.seq = self.seq.loc[self.seq > 0]
        elif sign == 'neg':
            self.seq = self.seq.loc[self.seq < 0]
        else:
            self.seq = self.seq.loc[self.seq != 0]

        self.dropna(inplace=True)
        #: (bool): verbose or not
        self.verbose = _deprecate_inform_(verbose, inform)
        if self.verbose:
            print(f"\nInitialized sequence with {len(self)} registries.")

        if sec_order:
            self.seq = subtract_sorted(self.seq.copy())
            self.dropna(inplace=True)
            self.reset_index(inplace=True)
            if verbose:
                print('Second Order Test. Initial series reduced '
                      f'to {len(self.seq)} entries.')

        ab = self.seq.abs()

        if self.seq.dtypes == 'int64':
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

    def mantissas(self, report=True, show_plot=True, figsize=(15, 8),
                  save_plot=None, save_plot_kwargs=None):
        """Calculates the mantissas, their mean and variance, and compares them
        with the mean and variance of a Benford's sequence.

        Args:
            report: prints the mamtissas mean, variance, skewness and kurtosis
                for the sequence studied, along with reference values.
            show_plot: plots the ordered mantissas and a line with the expected
                inclination. Defaults to True.
            figsize: tuple that sets the figure dimensions.
            save_plot: string with the path/name of the file in which the generated
                plot will be saved. Uses matplotlib.pyplot.savefig(). File format
                is infered by the file name extension. Only available when
                plot=True.
            save_plot_kwargs: dict with any of the kwargs accepted by
                matplotlib.pyplot.savefig()
                https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
                Only available when plot=True and save_plot is a string with the
                figure file path/name.
        """
        self['Mant'] = get_mantissas(self.seq.abs())
        if report:
            p = self[['seq', 'Mant']]
            p = p.loc[p.seq > 0].sort_values('Mant')
            print(f"The Mantissas MEAN is {p.Mant.mean()}. Ref: 0.5.")
            print(f"The Mantissas VARIANCE is {p.Mant.var()}. Ref: 0.083333.")
            print(f"The Mantissas SKEWNESS is {p.Mant.skew()}. \tRef: 0.")
            print(f"The Mantissas KURTOSIS is {p.Mant.kurt()}. \tRef: -1.2.")

        if show_plot:
            plot_ordered_mantissas(self.Mant, figsize=figsize,
                                   save_plot=save_plot, save_plot_kwargs=save_plot_kwargs)

    def first_digits(self, digs, confidence=None, high_Z='pos',
                     limit_N=None, MAD=False, MSE=False, chi_square=False,
                     KS=False, show_plot=True, save_plot=None, save_plot_kwargs=None,
                     simple=False, ret_df=False):
        """Performs the Benford First Digits test with the series of
        numbers provided, and populates the mapping dict for future
        selection of the original series.

        Args:
            digs: number of first digits to consider. Must be 1 (first digit),
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
            save_plot: string with the path/name of the file in which the generated
                plot will be saved. Uses matplotlib.pyplot.savefig(). File format
                is infered by the file name extension. Only available when
                plot=True.
            save_plot_kwargs: dict with any of the kwargs accepted by
                matplotlib.pyplot.savefig()
                https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
                Only available when plot=True and save_plot is a string with the
                figure file path/name.
            ret_df: returns the test DataFrame. Defaults to False. True if run by
                the test function.

        Returns:
            DataFrame with the Expected and Found proportions, and the Z scores of
                the differences
        """
        # Check on the possible values for confidence levels
        confidence = _check_confidence_(confidence)
        # Check on possible digits
        _check_test_(digs)

        temp = self.loc[self.ZN >= 10 ** (digs - 1)]
        temp[digs_dict[digs]] = (temp.ZN // 10 ** ((log10(temp.ZN).astype(
                                                   int)) - (digs - 1))).astype(
                                                       int)
        n, m = 10 ** (digs - 1), 10 ** (digs)
        x = arange(n, m)

        if simple:
            self.verbose = False
            show_plot = False
            df = prepare(temp[digs_dict[digs]], digs, limit_N=limit_N,
                         simple=True)
        else:
            N, df = prepare(temp[digs_dict[digs]], digs, limit_N=limit_N,
                            simple=False)

        if self.verbose:
            print(f"\nTest performed on {len(temp)} registries.\n"
                  f"Discarded {len(self) - len(temp)} records < {10 ** (digs - 1)}"
                  " after preparation.")
            if confidence is not None:
                _inform_(df, high_Z=high_Z, conf=confs[confidence])

        # Mean absolute difference
        if MAD:
            self.MAD = df.AbsDif.mean()
            if self.verbose:
                _report_mad_(digs, self.MAD)

        # Mean Square Error
        if MSE:
            self.MSE = (df.AbsDif ** 2).mean()

        # Chi-square statistic
        if chi_square:
            self.chi_square = chi_sq(df, ddf=len(df) - 1,
                                     confidence=confidence,
                                     verbose=self.verbose)
        # KS test
        if KS:
            self.KS = kolmogorov_smirnov(df, confidence=confidence, N=len(temp),
                                         verbose=self.verbose)

        # Plotting the expected frequncies (line) against the found ones(bars)
        if show_plot:
            plot_digs(df, x=x, y_Exp=df.Expected, y_Found=df.Found, N=N,
                       figsize=(2 * (digs ** 2 + 5), 1.5 * (digs ** 2 + 5)),
                       conf_Z=confs[confidence], save_plot=save_plot,
                       save_plot_kwargs=save_plot_kwargs)
        if ret_df:
            return df

    def second_digit(self, confidence=None, high_Z='pos',
                     limit_N=None, MAD=False, MSE=False, chi_square=False,
                     KS=False, show_plot=True, save_plot=None, save_plot_kwargs=None,
                     simple=False, ret_df=False):
        """Performs the Benford Second Digit test with the series of
        numbers provided.

        Args:
            verbose: tells the number of registries that are being subjected to
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
            save_plot: string with the path/name of the file in which the generated
                plot will be saved. Uses matplotlib.pyplot.savefig(). File format
                is infered by the file name extension. Only available when
                plot=True.
            save_plot_kwargs: dict with any of the kwargs accepted by
                matplotlib.pyplot.savefig()
                https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
                Only available when plot=True and save_plot is a string with the
                figure file path/name.
            ret_df: returns the test DataFrame. Defaults to False. True if run by
                the test function.

        Returns:
            DataFrame with the Expected and Found proportions, and the Z scores of
                the differences
        """
        confidence = _check_confidence_(confidence)

        conf = confs[confidence]

        temp = self.loc[self.ZN >= 10, :]
        temp['SD'] = (temp.ZN // 10 ** ((log10(temp.ZN)).astype(
                      int) - 1)) % 10

        if simple:
            self.verbose = False
            show_plot = False
            df = prepare(temp['SD'], 22, limit_N=limit_N, simple=True)
        else:
            N, df = prepare(temp['SD'], 22, limit_N=limit_N, simple=False)

        if self.verbose:
            print(f"\nTest performed on {len(temp)} registries.\nDiscarded "
                  f"{len(self) - len(temp)} records < 10 after preparation.")
            if confidence is not None:
                _inform_(df, high_Z, conf)

        # Mean absolute difference
        if MAD:
            self.MAD = df.AbsDif.mean()
            if self.verbose:
                _report_mad_(22, self.MAD)
        # Mean Square Error
        if MSE:
            self.MSE = (df.AbsDif ** 2).mean()

        # Chi-square statistic
        if chi_square:
            self.chi_square = chi_sq(df, ddf=9, confidence=confidence,
                                     verbose=self.verbose)
        # KS test
        if KS:
            self.KS = kolmogorov_smirnov(df, confidence=confidence, N=len(temp),
                                         verbose=self.verbose)

        # Plotting the expected frequncies (line) against the found ones(bars)
        if show_plot:
            plot_digs(df, x=arange(0, 10), y_Exp=df.Expected,
                       y_Found=df.Found, N=N, figsize=(10, 6), conf_Z=conf,
                       save_plot=save_plot, save_plot_kwargs=save_plot_kwargs)
        if ret_df:
            return df

    def last_two_digits(self, confidence=None, high_Z='pos',
                        limit_N=None, MAD=False, MSE=False, chi_square=False,
                        KS=False, show_plot=True, save_plot=None, save_plot_kwargs=None,
                        simple=False, ret_df=False):
        """Performs the Benford Last Two Digits test with the series of
        numbers provided.

        Args:
            verbose: tells the number of registries that are being subjected to
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
            save_plot: string with the path/name of the file in which the generated
                plot will be saved. Uses matplotlib.pyplot.savefig(). File format
                is infered by the file name extension. Only available when
                plot=True.
            save_plot_kwargs: dict with any of the kwargs accepted by
                matplotlib.pyplot.savefig()
                https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
                Only available when plot=True and save_plot is a string with the
                figure file path/name.
        
        Returns:
            DataFrame with the Expected and Found proportions, and the Z scores of
                the differences
        """
        confidence = _check_confidence_(confidence)
        conf = confs[confidence]

        temp = self.loc[self.ZN >= 1000]
        temp['L2D'] = temp.ZN % 100

        if simple:
            self.verbose = False
            show_plot = False
            df = prepare(temp['L2D'], -2, limit_N=limit_N, simple=True)
        else:
            N, df = prepare(temp['L2D'], -2, limit_N=limit_N, simple=False)

        if self.verbose:
            print(f"\nTest performed on {len(temp)} registries.\n\nDiscarded "
                  f"{len(self) - len(temp)} records < 1000 after preparation")
            if confidence is not None:
                _inform_(df, high_Z, conf)

        # Mean absolute difference
        if MAD:
            self.MAD = df.AbsDif.mean()
            if self.verbose:
                _report_mad_(-2, self.MAD)
        # Mean Square Error
        if MSE:
            self.MSE = (df.AbsDif ** 2).mean()

        # Chi-square statistic
        if chi_square:
            self.chi_square = chi_sq(df, ddf=99, confidence=confidence,
                                     verbose=self.verbose)
        # KS test
        if KS:
            self.KS = kolmogorov_smirnov(df, confidence=confidence, N=len(temp),
                                         verbose=self.verbose)

        # Plotting expected frequencies (line) versus found ones (bars)
        if show_plot:
            plot_digs(df, x=arange(0, 100), y_Exp=df.Expected,
                       y_Found=df.Found, N=N, figsize=(15, 5),
                       conf_Z=conf, text_x=True, save_plot=save_plot,
                       save_plot_kwargs=save_plot_kwargs)
        if ret_df:
            return df

    def summation(self, digs=2, top=20, show_plot=True, save_plot=None,
                  save_plot_kwargs=None, ret_df=False):
        """Performs the Summation test. In a Benford series, the sums of the
        entries begining with the same digits tends to be the same.

        Args:
            digs: tells the first digits to use. 1- first; 2- first two;
                3- first three. Defaults to 2.
            top: choses how many top values to show. Defaults to 20.
            show_plot: plots the results. Defaults to True.
            save_plot: string with the path/name of the file in which the generated
                plot will be saved. Uses matplotlib.pyplot.savefig(). File format
                is infered by the file name extension. Only available when
                plot=True.
            save_plot_kwargs: dict with any of the kwargs accepted by
                matplotlib.pyplot.savefig()
                https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
                Only available when plot=True and save_plot is a string with the
                figure file path/name.
        
        Returns:
            DataFrame with the Expected and Found proportions, and their
                absolute differences
        """
        _check_digs_(digs)

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
        df = df[['Summ', 'Percent']]
        df['AbsDif'] = (df.Percent - li).abs()

        if self.verbose:
            # N = len(self)
            print(f"\nTest performed on {len(self)} registries.\n")
            print(f"The top {top} diferences are:\n")
            print(df[:top])

        if show_plot:
            plot_sum(df, figsize=(
                       2 * (digs ** 2 + 5), 1.5 * (digs ** 2 + 5)), li=li,
                       save_plot=save_plot, save_plot_kwargs=save_plot_kwargs)

        if ret_df:
            return df

    def duplicates(self, top_Rep=20, inform=None):
        """Performs a duplicates test and maps the duplicates count in descending
        order.

        Args:
            verbose: tells how many duplicated entries were found and prints the
                top numbers according to the top_Rep argument. Defaluts to True.
            top_Rep: int or None. Chooses how many duplicated entries will be
                shown withe the top repititions. Defaluts to 20. If None, returns
                al the ordered repetitions.

        Returns:
            DataFrame with the duplicated records and their occurrence counts,
                in descending order (if verbose is False; if True, prints to
                terminal).

        Raises:
            ValueError: if the `top_Rep` arg is not int or None.
        """
        if top_Rep is not None and not isinstance(top_Rep, int):
            raise ValueError('The top_Rep argument must be an int or None.')

        dup = self[['seq']][self.seq.duplicated(keep=False)]
        dup_count = dup.groupby(self.seq).count()

        dup_count.index.names = ['Entries']
        dup_count.rename(columns={'seq': 'Count'}, inplace=True)

        dup_count.sort_values('Count', ascending=False, inplace=True)

        # self.maps['dup'] = dup_count.index[:top_Rep].values  # array

        if self.verbose:
            print(f'\nFound {len(dup_count)} duplicated entries.\n'
                  f'The entries with the {top_Rep} highest repitition counts are:')
            print(dup_count.head(top_Rep))
        else:
            return dup_count


class Mantissas(object):
    """
    Returns a Series with the data mantissas,

    Args:
        data: sequence to compute mantissas from, numpy 1D array, pandas
            Series of pandas DataFrame column.
    Attributes:
        data (DataFrame): holds the computed mantissas and, if the arc_test
            is also called, the respecttive x and Y coordinates for the plot.
        stats (dict): holds the relevant statistics about the data mantissas.
    """

    def __init__(self, data):

        data = Series(_check_num_array_(data))
        data = data.dropna().loc[data != 0].abs()

        self.data = DataFrame({'Mantissa': get_mantissas(data.abs())})

        self.stats = {'Mean': self.data.Mantissa.mean(),
                      'Var': self.data.Mantissa.var(),
                      'Skew': self.data.Mantissa.skew(),
                      'Kurt': self.data.Mantissa.kurt()}

    def report(self, show_plot=True, save_plot=None, save_plot_kwargs=None):
        """Displays the Mantissas stats.

        Args:
            show_plot: shows the ordered mantissas plot and the Arc Test plot.
                Defaults to True.
            save_plot: string with the path/name of the file in which the generated
                plot will be saved. Uses matplotlib.pyplot.savefig(). File format
                is infered by the file name extension. Only available when
                plot=True.
            save_plot_kwargs: dict with any of the kwargs accepted by
                matplotlib.pyplot.savefig()
                https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
                Only available when plot=True and save_plot is a string with the
                figure file path/name.
        """
        print("\n", '  Mantissas Test  '.center(52, '#'))
        print(f"\nThe Mantissas MEAN is      {self.stats['Mean']:.6f}."
              "\tRef: 0.5")
        print(f"The Mantissas VARIANCE is  {self.stats['Var']:.6f}."
              "\tRef: 0.08333")
        print(f"The Mantissas SKEWNESS is  {self.stats['Skew']:.6f}."
              "\tRef: 0.0")
        print(f"The Mantissas KURTOSIS is  {self.stats['Kurt']:.6f}."
              "\tRef: -1.2\n")
        if show_plot:
            self.show_plot(save_plot=save_plot, save_plot_kwargs=save_plot_kwargs)
            self.arc_test(save_plot=save_plot, save_plot_kwargs=save_plot_kwargs)

    def show_plot(self, figsize=(12, 12), save_plot=None, save_plot_kwargs=None):
        """Plots the ordered mantissas and compares them to the expected, straight
        line that should be formed in a Benford-cmpliant set.

        Args:
            figsize: tuple that sets the figure size.
            save_plot: string with the path/name of the file in which the generated
                plot will be saved. Uses matplotlib.pyplot.savefig(). File format
                is infered by the file name extension.
            save_plot_kwargs: dict with any of the kwargs accepted by
                matplotlib.pyplot.savefig()
                https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
                Only available when save_plot is a string with the figure file
                path/name.
        """
        plot_ordered_mantissas(self.data.Mantissa, figsize=figsize,
                               save_plot=save_plot, save_plot_kwargs=save_plot_kwargs)
 
    def arc_test(self, grid=True, figsize=12, save_plot=None, save_plot_kwargs=None):
        """
        Add two columns to Mantissas's DataFrame equal to their "X" and "Y"
        coordinates, plots its to a scatter plot and calculates the gravity
        center of the circle.

        Args:
            grid:show grid of the plot. Defaluts to True.
            figsize: size of the figure to be displayed. Since it is a square,
                there is no need to provide a tuple, like is usually the case with
                matplotlib.
            save_plot: string with the path/name of the file in which the generated
                plot will be saved. Uses matplotlib.pyplot.savefig(). File format
                is infered by the file name extension.
            save_plot_kwargs: dict with any of the kwargs accepted by
                matplotlib.pyplot.savefig()
                https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
                Only available when save_plot is a string with the figure file
                path/name.
        """
        if self.stats.get('gravity_center') is None:
            self.data['mant_x'] = cos(2 * pi * self.data.Mantissa)
            self.data['mant_y'] = sin(2 * pi * self.data.Mantissa)
            self.stats['gravity_center'] = (self.data.mant_x.mean(),
                                            self.data.mant_y.mean())
        plot_mantissa_arc_test(self.data, self.stats['gravity_center'],
                               figsize=figsize, save_plot=save_plot,
                               save_plot_kwargs=save_plot_kwargs)


class Roll_mad(object):
    """Applies the MAD to sequential subsets of the Series, returning another
    Series.

    Args:
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

    """

    def __init__(self, data, test, window, decimals=2, sign='all'):

        #: the test (F1D, SD, F2D...) used for the MAD calculation and critical values
        self.test = _check_test_(test)

        if not isinstance(data, Source):
            data = Source(data, sign=sign, decimals=decimals, verbose=False)

        Exp, ind = prep_to_roll(data, self.test)

        self.roll_series = data[digs_dict[test]].rolling(
                                window=window).apply(mad_to_roll, 
                                    args=(Exp, ind), raw=False)
        self.roll_series.dropna(inplace=True)

    def show_plot(self, figsize=(15, 8), save_plot=None, save_plot_kwargs=None):
        """Shows the rolling MAD plot

        Args:
            figsize: the figure dimensions.
            save_plot: string with the path/name of the file in which the generated
                plot will be saved. Uses matplotlib.pyplot.savefig(). File format
                is infered by the file name extension.
            save_plot_kwargs: dict with any of the kwargs accepted by
                matplotlib.pyplot.savefig()
                https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
                Only available when save_plot is a string with the figure file
                path/name.
        """
        plot_roll_mad(self, figsize=figsize,
                      save_plot=save_plot, save_plot_kwargs=save_plot_kwargs)


class Roll_mse(object):
    """Applies the MSE to sequential subsets of the Series, returning another
    Series.

    Args:
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
        sign: tells which portion of the data to consider. 'pos': only the positive
            entries; 'neg': only negative entries; 'all': all entries but zeros.
            Defaults to 'all'.
    """

    def __init__(self, data, test, window, decimals=2, sign='all'):

        test = _check_test_(test)

        if not isinstance(data, Source):
            data = Source(data, sign=sign, decimals=decimals, verbose=False)

        Exp, ind = prep_to_roll(data, test)

        self.roll_series = data[digs_dict[test]].rolling(
                                window=window).apply(mse_to_roll, 
                                    args=(Exp, ind), raw=False)
        self.roll_series.dropna(inplace=True)

    def show_plot(self, figsize=(15, 8), save_plot=None, save_plot_kwargs=None):
        """Shows the rolling MSE plot

        Args:
            figsize: the figure dimensions.
            save_plot: string with the path/name of the file in which the generated
                plot will be saved. Uses matplotlib.pyplot.savefig(). File format
                is infered by the file name extension.
            save_plot_kwargs: dict with any of the kwargs accepted by
                matplotlib.pyplot.savefig()
                https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
                Only available when save_plot is a string with the figure file
                path/name.
        """
        plot_roll_mse(self.roll_series, figsize=figsize,
                      save_plot=save_plot, save_plot_kwargs=save_plot_kwargs)


def first_digits(data, digs, decimals=2, sign='all', verbose=True,
                 confidence=None, high_Z='pos', limit_N=None,
                 MAD=False, MSE=False, chi_square=False, KS=False,
                 show_plot=True, save_plot=None, save_plot_kwargs=None,
                 inform=None):
    """Performs the Benford First Digits test on the series of
    numbers provided.

    Args:
        data: sequence of numbers to be evaluated. Must be a numpy 1D array,
            a pandas Series or a pandas DataFrame column, with values being
            integers or floats.
        decimals: number of decimal places to consider. Defaluts to 2.
            If integers, set to 0. If set to -infer-, it will remove the zeros
            and consider up to the fifth decimal place to the right, but will
            loose performance.
        sign: tells which portion of the data to consider. 'pos': only the positive
            entries; 'neg': only negative entries; 'all': all entries but zeros.
            Defaults to 'all'.
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
        save_plot: string with the path/name of the file in which the generated
            plot will be saved. Uses matplotlib.pyplot.savefig(). File format
            is infered by the file name extension. Only available when
            plot=True.
        save_plot_kwargs: dict with any of the kwargs accepted by
            matplotlib.pyplot.savefig()
            https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
            Only available when plot=True and save_plot is a string with the
            figure file path/name.
    
    Returns:
        DataFrame with the Expected and Found proportions, and the Z scores of
            the differences if the confidence is not None.
    """
    verbose = _deprecate_inform_(verbose, inform)

    if not isinstance(data, Source):
        data = Source(data, decimals=decimals, sign=sign, verbose=verbose)

    data = data.first_digits(digs, confidence=confidence, high_Z=high_Z,
                             limit_N=limit_N, MAD=MAD, MSE=MSE,
                             chi_square=chi_square, KS=KS, show_plot=show_plot,
                             save_plot=save_plot, save_plot_kwargs=save_plot_kwargs,
                             ret_df=True)

    if confidence is not None:
        data = data[['Counts', 'Found', 'Expected', 'Z_score']]
        return data.sort_values('Z_score', ascending=False)
    else:
        return data[['Counts', 'Found', 'Expected']]


def second_digit(data, decimals=2, sign='all', verbose=True,
                 confidence=None, high_Z='pos', limit_N=None,
                 MAD=False, MSE=False, chi_square=False, KS=False,
                 show_plot=True, save_plot=None, save_plot_kwargs=None,
                 inform=None):
    """Performs the Benford Second Digits test on the series of
    numbers provided.

    Args:
        data: sequence of numbers to be evaluated. Must be a numpy 1D array,
            a pandas Series or a pandas DataFrame column, with values being
            integers or floats.
        decimals: number of decimal places to consider. Defaluts to 2.
            If integers, set to 0. If set to -infer-, it will remove the zeros
            and consider up to the fifth decimal place to the right, but will
            loose performance.
        sign: tells which portion of the data to consider. 'pos': only the positive
            entries; 'neg': only negative entries; 'all': all entries but zeros.
            Defaults to 'all'.
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
        save_plot: string with the path/name of the file in which the generated
            plot will be saved. Uses matplotlib.pyplot.savefig(). File format
            is infered by the file name extension. Only available when
            plot=True.
        save_plot_kwargs: dict with any of the kwargs accepted by
            matplotlib.pyplot.savefig()
            https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
            Only available when plot=True and save_plot is a string with the
            figure file path/name.

    Returns:
        DataFrame with the Expected and Found proportions, and the Z scores of
            the differences if the confidence is not None.
    """
    verbose = _deprecate_inform_(verbose, inform)

    if not isinstance(data, Source):
        data = Source(data, sign=sign, decimals=decimals, verbose=verbose)

    data = data.second_digit(confidence=confidence, high_Z=high_Z,
                             limit_N=limit_N, MAD=MAD, MSE=MSE,
                             chi_square=chi_square, KS=KS, show_plot=show_plot,
                             save_plot=save_plot, save_plot_kwargs=save_plot_kwargs,
                             ret_df=True)
    if confidence is not None:
        data = data[['Counts', 'Found', 'Expected', 'Z_score']]
        return data.sort_values('Z_score', ascending=False)
    else:
        return data[['Counts', 'Found', 'Expected']]


def last_two_digits(data, decimals=2, sign='all', verbose=True,
                    confidence=None, high_Z='pos', limit_N=None,
                    MAD=False, MSE=False, chi_square=False, KS=False,
                    show_plot=True, save_plot=None, save_plot_kwargs=None,
                    inform=None):
    """Performs the Last Two Digits test on the series of
    numbers provided.

    Args:
        data: sequence of numbers to be evaluated. Must be a numpy 1D array,
            a pandas Series or a pandas DataFrame column,with values being
            integers or floats.
        decimals: number of decimal places to consider. Defaluts to 2.
            If integers, set to 0. If set to -infer-, it will remove the zeros
            and consider up to the fifth decimal place to the right, but will
            loose performance.
        sign: tells which portion of the data to consider. 'pos': only the positive
            entries; 'neg': only negative entries; 'all': all entries but zeros.
            Defaults to 'all'.
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
        save_plot: string with the path/name of the file in which the generated
            plot will be saved. Uses matplotlib.pyplot.savefig(). File format
            is infered by the file name extension. Only available when
            plot=True.
        save_plot_kwargs: dict with any of the kwargs accepted by
            matplotlib.pyplot.savefig()
            https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
            Only available when plot=True and save_plot is a string with the
            figure file path/name.

    Returns:
        DataFrame with the Expected and Found proportions, and the Z scores of
            the differences if the confidence is not None.
    """
    verbose = _deprecate_inform_(verbose, inform)

    if not isinstance(data, Source):
        data = Source(data, decimals=decimals, sign=sign, verbose=verbose)

    data = data.last_two_digits(confidence=confidence, high_Z=high_Z,
                                limit_N=limit_N, MAD=MAD,
                                MSE=MSE, chi_square=chi_square, KS=KS,
                                show_plot=show_plot, save_plot=save_plot,
                                save_plot_kwargs=save_plot_kwargs, ret_df=True)

    if confidence is not None:
        data = data[['Counts', 'Found', 'Expected', 'Z_score']]
        return data.sort_values('Z_score', ascending=False)
    else:
        return data[['Counts', 'Found', 'Expected']]


def mantissas(data, report=True, show_plot=True, arc_test=True,
              save_plot=None, save_plot_kwargs=None, inform=None):
    """Extraxts the mantissas of the records logarithms

    Args:
        data: sequence to compute mantissas from, numpy 1D array, pandas Series
            of pandas DataFrame column.
        report: prints the mamtissas mean, variance, skewness and kurtosis
            for the sequence studied, along with reference values.
        show_plot: plots the ordered mantissas and a line with the expected
            inclination. Defaults to True.
        arc_test: draws the Arc Test plot. Defaluts to True.
        save_plot: string with the path/name of the file in which the generated
            plot will be saved. Uses matplotlib.pyplot.savefig(). File format
            is infered by the file name extension. Only available when
            plot=True.
        save_plot_kwargs: dict with any of the kwargs accepted by
            matplotlib.pyplot.savefig()
            https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
            Only available when plot=True and save_plot is a string with the
            figure file path/name.
    
    Returns:
        Series with the data mantissas.
    """
    report = _deprecate_inform_(report, inform)

    mant = Mantissas(data)
    if report:
        mant.report(save_plot=save_plot, save_plot_kwargs=save_plot_kwargs)
    if show_plot:
        mant.show_plot(save_plot=save_plot, save_plot_kwargs=save_plot_kwargs)
    if arc_test:
        mant.arc_test(save_plot=save_plot, save_plot_kwargs=save_plot_kwargs)
    return mant


def summation(data, digs=2, decimals=2, sign='all', top=20, verbose=True,
              show_plot=True, save_plot=None, save_plot_kwargs=None, inform=None):
    """Performs the Summation test. In a Benford series, the sums of the
    entries begining with the same digits tends to be the same.
    Works only with the First Digits (1, 2 or 3) test.

    Args:
        digs: tells the first digits to use: 1- first; 2- first two;
            3- first three. Defaults to 2.
        decimals: number of decimal places to consider. Defaluts to 2.
            If integers, set to 0. If set to -infer-, it will remove the zeros
            and consider up to the fifth decimal place to the right, but will
            loose performance.
        top: choses how many top values to show. Defaults to 20.
        show_plot: plots the results. Defaults to True.
        save_plot: string with the path/name of the file in which the generated
            plot will be saved. Uses matplotlib.pyplot.savefig(). File format
            is infered by the file name extension. Only available when
            plot=True.
        save_plot_kwargs: dict with any of the kwargs accepted by
            matplotlib.pyplot.savefig()
            https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
            Only available when plot=True and save_plot is a string with the
            figure file path/name.
    
    Returns:
        DataFrame with the Summation test, whether sorted in descending order
            (if verbose == True) or not.
    """
    verbose = _deprecate_inform_(verbose, inform)

    if not isinstance(data, Source):
        data = Source(data, sign=sign, decimals=decimals, verbose=verbose)

    data = data.summation(digs=digs, top=top,
                          show_plot=show_plot, save_plot=save_plot,
                          save_plot_kwargs=save_plot_kwargs, ret_df=True)
    if verbose:
        return data.sort_values('AbsDif', ascending=False)
    else:
        return data


def mad(data, test, decimals=2, sign='all', verbose=False):
    """Calculates the Mean Absolute Deviation of the Series

    Args:
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
            Defaults to all.
    Returns:
        float: the Mean Absolute Deviation of the Series
    """
    data = _check_num_array_(data)
    test = _check_test_(test)
    start = Source(data, sign=sign, decimals=decimals, verbose=verbose)
    if test in [1, 2, 3]:
        start.first_digits(digs=test, MAD=True, MSE=True, simple=True)
    elif test == 22:
        start.second_digit(MAD=True, MSE=False, simple=True)
    else:
        start.last_two_digits(MAD=True, MSE=False, simple=True)
    return start.MAD


def mse(data, test, decimals=2, sign='all', verbose=False):
    """Calculates the Mean Squared Error of the Series

    Args:
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
            Defaults to all.
    Returns:
        float: the Mean Squared Error of the Series
    """
    data = _check_num_array_(data)
    test = _check_test_(test)
    start = Source(data, sign=sign, decimals=decimals, verbose=verbose)
    if test in [1, 2, 3]:
        start.first_digits(digs=test, MAD=False, MSE=True, simple=True)
    elif test == 22:
        start.second_digit(MAD=False, MSE=True, simple=True)
    else:
        start.last_two_digits(MAD=False, MSE=True, simple=True)
    return start.MSE


def mad_summ(data, test, decimals=2, sign='all', verbose=False):
    """Calculate the Mean Absolute Deviation of the Summation Test

    Args:
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
            Defaults to all.

    Returns:
        float: the Mean Absolute Deviation of the Summation Test
    """
    data = _check_num_array_(data)
    test = _check_digs_(test)

    start = Source(data, sign=sign, decimals=decimals, verbose=verbose)
    temp = start.loc[start.ZN >= 10 ** (test - 1)]
    temp[digs_dict[test]] = (temp.ZN // 10 ** ((log10(temp.ZN).astype(
                                                int)) - (test - 1))).astype(
                                                    int)
    li = 1. / (9 * (10 ** (test - 1)))

    df = temp.groupby(digs_dict[test]).sum()
    return mean(abs(df.ZN / df.ZN.sum() - li))


def rolling_mad(data, test, window, decimals=2, sign='all',
                show_plot=False, save_plot=None, save_plot_kwargs=None):
    """Applies the MAD to sequential subsets of the records.

    Args:
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
        save_plot: string with the path/name of the file in which the generated
            plot will be saved. Uses matplotlib.pyplot.savefig(). File format
            is infered by the file name extension. Only available when
            plot=True.
        save_plot_kwargs: dict with any of the kwargs accepted by
            matplotlib.pyplot.savefig()
            https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
            Only available when plot=True and save_plot is a string with the
            figure file path/name.
    
    Returns:
        Series with sequentially computed MADs.
    """
    data = _check_num_array_(data)
    r_mad = Roll_mad(data, test, window, decimals, sign)
    if show_plot:
        r_mad.show_plot(save_plot=save_plot, save_plot_kwargs=save_plot_kwargs)
    return r_mad.roll_series


def rolling_mse(data, test, window, decimals=2, sign='all',
                show_plot=False, save_plot=None, save_plot_kwargs=None):
    """Applies the MSE to sequential subsets of the records.

    Args:
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
        save_plot: string with the path/name of the file in which the generated
            plot will be saved. Uses matplotlib.pyplot.savefig(). File format
            is infered by the file name extension. Only available when
            plot=True.
        save_plot_kwargs: dict with any of the kwargs accepted by
            matplotlib.pyplot.savefig()
            https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
            Only available when plot=True and save_plot is a string with the
            figure file path/name.
    
    Returns:
        Series with sequentially computed MSEs.
    """
    data = _check_num_array_(data)
    r_mse = Roll_mse(data, test, window, decimals, sign)
    if show_plot:
        r_mse.show_plot(save_plot=save_plot, save_plot_kwargs=save_plot_kwargs)
    return r_mse.roll_series


def duplicates(data, top_Rep=20, verbose=True, inform=None):
    """Performs a duplicates test and maps the duplicates count in descending
    order.

    Args:
        data: sequence to take the duplicates from. pandas Series or
            numpy Ndarray.
        verbose: tells how many duplicated entries were found and prints the
            top numbers according to the top_Rep argument. Defaluts to True.
        top_Rep: chooses how many duplicated entries will be
            shown withe the top repititions. int or None. Defaluts to 20.
            If None, returns al the ordered repetitions.

    Returns:
        DataFrame with the duplicated records and their respective counts

    Raises:
        ValueError: if the `top_Rep` arg is not int or None.
    """
    verbose = _deprecate_inform_(verbose, inform)

    if top_Rep is not None and not isinstance(top_Rep, int):
        raise ValueError('The top_Rep argument must be an int or None.')

    if not isinstance(data, Series):
        try:
            data = Series(data)
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
                 show_plot=True, save_plot=None, save_plot_kwargs=None, inform=None):
    """Performs the chosen test after subtracting the ordered sequence by itself.
    Hence Second Order.

    Args:
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
            Defaults to all.
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
        save_plot: string with the path/name of the file in which the generated
            plot will be saved. Uses matplotlib.pyplot.savefig(). File format
            is infered by the file name extension. Only available when
            plot=True.
        save_plot_kwargs: dict with any of the kwargs accepted by
            matplotlib.pyplot.savefig()
            https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html
            Only available when plot=True and save_plot is a string with the
            figure file path/name.
    
    Returns:
        DataFrame of the test chosen, but applied on Second Order pre-
            processed data.
    """
    test = _check_test_(test)

    verbose = _deprecate_inform_(verbose, inform)

    data = Source(data, decimals=decimals, sign=sign,
                  sec_order=True, verbose=verbose)
    if test in [1, 2, 3]:
        data.first_digits(digs=test, MAD=MAD,
                          confidence=confidence, high_Z=high_Z,
                          limit_N=limit_N, MSE=MSE, show_plot=show_plot,
                          save_plot=save_plot, save_plot_kwargs=save_plot_kwargs)
    elif test == 22:
        data.second_digit(MAD=MAD, confidence=confidence, high_Z=high_Z,
                          limit_N=limit_N, MSE=MSE, show_plot=show_plot,
                          save_plot=save_plot, save_plot_kwargs=save_plot_kwargs)
    else:
        data.last_two_digits(MAD=MAD, confidence=confidence, high_Z=high_Z,
                             limit_N=limit_N, MSE=MSE, show_plot=show_plot,
                             save_plot=save_plot, save_plot_kwargs=save_plot_kwargs)
    return data
