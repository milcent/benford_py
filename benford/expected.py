from pandas import DataFrame
from numpy import array, arange, log10
from .checks import _check_digs_
from .viz import plot_expected


class First(DataFrame):
    """Holds the expected probabilities of the First, First Two, or
    First Three digits according to Benford's distribution.

    Args:
        digs: 1, 2 or 3 - tells which of the first digits to consider:
            1 for the First Digit, 2 for the First Two Digits and 3 for
            the First Three Digits.
        plot: option to plot a bar chart of the Expected proportions.
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

    def __init__(self, digs, plot=True, save_plot=None, save_plot_kwargs=None):
        _check_digs_(digs)
        dig_name = f'First_{digs}_Dig'
        exp_array, dig_array = _gen_first_digits_(digs)
 
        DataFrame.__init__(self, {'Expected': exp_array}, index=dig_array)
        self.index.names = [dig_name]

        if plot:
            plot_expected(self, digs, save_plot=save_plot,
                          save_plot_kwargs=save_plot_kwargs)


class Second(DataFrame):
    """Holds the expected probabilities of the Second Digits
    according to Benford's distribution.

    Args:
        plot: option to plot a bar chart of the Expected proportions.
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
    def __init__(self, plot=True, save_plot=None, save_plot_kwargs=None):

        exp, sec_digs = _gen_second_digits_()

        DataFrame.__init__(self, {'Expected': exp, 'Sec_Dig': sec_digs})
        self.set_index("Sec_Dig", inplace=True)

        if plot:
            plot_expected(self, 22, save_plot=save_plot,
                          save_plot_kwargs=save_plot_kwargs)


class LastTwo(DataFrame):
    """Holds the expected probabilities of the Last Two Digits
    according to Benford's distribution.

    Args:
        plot: option to plot a bar chart of the Expected proportions.
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
    def __init__(self, num=False, plot=True, save_plot=None, save_plot_kwargs=None):
        exp, l2d = _gen_last_two_digits_(num=num)
        DataFrame.__init__(self, {'Expected': exp,
                                  'Last_2_Dig': l2d})
        self.set_index('Last_2_Dig', inplace=True)
        if plot:
            plot_expected(self, -2, save_plot=save_plot,
                          save_plot_kwargs=save_plot_kwargs)


def _get_expected_digits_(digs):
    """Chooses the Exxpected class to be used in a test

    Args:
        digs: the int corresponding to the Expected class to be instantiated

    Returns:
        the Expected instance forthe propoer test to be performed
    """
    if digs in [1, 2, 3]:
        return First(digs, plot=False)
    elif digs == 22:
        return Second(plot=False)
    else:
        return LastTwo(num=True, plot=False)


def _gen_last_two_digits_(num=False):
    """Creates two arrays, one with the possible last two digits and one with
    thei respective probabilities

    Args:
        num: returns numeric (ints) values. Defaluts to False,
            which returns strings.

    Returns:
        exp (np.array): Array with the (constant) probabilities of occurrence of
            each pair of last two digits 
        l2d (np.array): Array of ints or str, in any case representing all 100
            possible combinations of last two digits
    """
    exp = array([1 / 99.] * 100)
    l2d = arange(0, 100)
    if num:
        return exp, l2d
    l2d = l2d.astype(str)
    l2d[:10] = array(['00', '01', '02', '03', '04', '05',
                    '06', '07', '08', '09'])
    return exp, l2d

def _gen_first_digits_(digs):
    """Creates two arrays, one with the possible digits combinations and the
    other with their respective expected probabilities according to Benford

    Args:
        digs (int): 1, 2 or 3, for generation of the first, first two, or first
            three digits

    Returns:
        (tuple of arrays): the expected probabilities array and the digits
            combination array. 
    """
    dig_array = arange(10 ** (digs - 1), 10 ** digs)
    exp_prob = log10(1 + (1. / dig_array))
    return exp_prob, dig_array

def _gen_second_digits_():
    """Creates two arrays, one with he possible second digits combinations and
    the other with their respective expected probabilities according to Benford

    Returns:
        (tuple of arrays): the expected probabilities array and the second
        digits array.
    """
    exp_f2d, _ = _gen_first_digits_(2)
    sec_digs = range(10)
    sec_digs_in_f2d = array(list(range(10)) * 9)
    exp = array([exp_f2d[sec_digs_in_f2d == i].sum() for i in sec_digs])
    return exp, array(sec_digs)