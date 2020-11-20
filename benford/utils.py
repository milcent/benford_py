from pandas import Series, DataFrame
from numpy import array, arange, log10, ndarray
from .expected import _test_
from .constants import digs_dict, rev_digs
from .stats import Z_score
from .checks import _check_num_array_, _check_sign_, _check_decimals_


def _set_N_(len_df, limit_N):
    """"""
    # Assigning to N the superior limit or the lenght of the series
    if limit_N is None or limit_N > len_df:
        return max(1, len_df)
    # Check on limit_N being a positive integer
    else:
        if limit_N < 0 or not isinstance(limit_N, int):
            raise ValueError("limit_N must be None or a positive integer.")
        else:
            return max(1, limit_N)


def get_mantissas(arr):
    """Computes the mantissas, the non-integer part of the log of a number.

    Args:
        arr: array of integers or floats

    Returns:
        Array of floats withe logs mantissas
    """
    log_a = abs(log10(arr))
    return log_a - log_a.astype(int)  # the number - its integer part


def input_data(given):
    """Internalizes and transforms the input data

    Args:
        given: ndarray, Series or tuple with DataFrame and name of the
            column to analyze

    Returns:
        The raw inputed data and the result of its first pre-processing,
            when required.
    """
    if type(given) == Series:
        data = chosen = given
    elif type(given) == ndarray:
        data = given
        chosen = Series(given)
    elif type(given) == tuple:
        if (type(given[0]) != DataFrame) | (type(given[1]) != str):
            raise TypeError('The data tuple must be composed of a pandas '
                            'DataFrame and the name (str) of the chosen '
                            'column, in that order')
        data = given[0]
        chosen = given[0][given[1]]
    else:
        raise TypeError("Wrong data input type. Check docstring.")
    return data, chosen


def set_sign(data, sign="all"):
    """
    """
    sign = _check_sign_(sign)

    if sign == 'all':
        data.seq = data.seq.loc[data.seq != 0]
    elif sign == 'pos':
        data.seq = data.seq.loc[data.seq > 0]
    else:
        data.seq = data.seq.loc[data.seq < 0]

    return data.dropna()


def get_times_10_power(data, decimals=2):
    """"""
    decimals = _check_decimals_(decimals)

    ab = data.seq.abs()

    if data.seq.dtypes == 'int64':
        data['ZN'] = ab
    else:
        if decimals == 'infer':
            data['ZN'] = ab.astype(str).str\
                .replace('.', '')\
                .str.lstrip('0')\
                .str[:5].astype(int)
        else:
            data['ZN'] = (ab * (10 ** decimals)).astype(int)
    return data


def get_digs(data, decimals=2, sign="all"):
    """ 
    """
    df = DataFrame({'seq': _check_num_array_(data)})

    df = set_sign(df, sign=sign)

    df = get_times_10_power(df, decimals=decimals)

    # First digits
    for col in ['F1D', 'F2D', 'F3D']:
        temp = df.ZN.loc[df.ZN >= 10 ** (rev_digs[col] - 1)]
        df[col] = (temp // 10 ** ((log10(temp).astype(int)) -
                                  (rev_digs[col] - 1)))
        # fill NANs with -1, which is a non-usable value for digits,
        # to be discarded later.
        df[col] = df[col].fillna(-1).astype(int)
    # Second digit
    temp_sd = df.loc[df.ZN >= 10]
    df['SD'] = (temp_sd.ZN // 10**((log10(temp_sd.ZN)).astype(int) -
                                   1)) % 10
    df['SD'] = df['SD'].fillna(-1).astype(int)
    # Last two digits
    temp_l2d = df.loc[df.ZN >= 1000]
    df['L2D'] = temp_l2d.ZN % 100
    df['L2D'] = df['L2D'].fillna(-1).astype(int)
    return df


def get_proportions(data):
    """
    """
    counts = data.value_counts()
    # get their relative frequencies
    proportions = data.value_counts(normalize=True)
    # crate dataframe from them
    return DataFrame({'Counts': counts, 'Found': proportions}).sort_index()


def join_expect_found_diff(data, digs):
    """
    """
    dd = _test_(digs).join(data).fillna(0)
    # create column with absolute differences
    dd['Dif'] = dd.Found - dd.Expected
    dd['AbsDif'] = dd.Dif.abs()
    return dd


def prepare(data, digs, limit_N=None, simple=False):
    """Transforms the original number sequence into a DataFrame reduced
    by the ocurrences of the chosen digits, creating other computed
    columns
    """
    df = get_proportions(data)
    dd = join_expect_found_diff(df, digs)
    if simple:
        del dd['Dif']
        return dd
    else:
        N = _set_N_(len(data), limit_N=limit_N)
        dd['Z_score'] = Z_score(dd, N)
        return N, dd


def subtract_sorted(data):
    """Subtracts the sorted sequence elements from each other, discarding zeros.
    Used in the Second Order test
    """
    temp = data.copy().sort_values(ignore_index=True)
    temp = (temp - temp.shift(1)).dropna()
    return temp.loc[temp != 0]


def prep_to_roll(start, test):
    """Used by the rolling mad and rolling mean, prepares each test and
    respective expected proportions for later application to the Series subset
    """
    if test in [1, 2, 3]:
        start[digs_dict[test]] = start.ZN // 10 ** ((
            log10(start.ZN).astype(int)) - (test - 1))
        start = start.loc[start.ZN >= 10 ** (test - 1)]

        ind = arange(10 ** (test - 1), 10 ** test)
        Exp = log10(1 + (1. / ind))

    elif test == 22:
        start[digs_dict[test]] = (start.ZN // 10 ** ((
            log10(start.ZN)).astype(int) - 1)) % 10
        start = start.loc[start.ZN >= 10]

        Expec = log10(1 + (1. / arange(10, 100)))
        temp = DataFrame({'Expected': Expec, 'Sec_Dig':
                          array(list(range(10)) * 9)})
        Exp = temp.groupby('Sec_Dig').sum().values.reshape(10,)
        ind = arange(0, 10)

    else:
        start[digs_dict[test]] = start.ZN % 100
        start = start.loc[start.ZN >= 1000]

        ind = arange(0, 100)
        Exp = array([1 / 99.] * 100)

    return Exp, ind


def mad_to_roll(arr, Exp, ind):
    """Mean Absolute Deviation used in the rolling function
    """
    prop = arr.value_counts(normalize=True).sort_index()

    if len(prop) < len(Exp):
        prop = prop.reindex(ind).fillna(0)

    return abs(prop - Exp).mean()


def mse_to_roll(arr, Exp, ind):
    """Mean Squared Error used in the rolling function
    """
    temp = arr.value_counts(normalize=True).sort_index()

    if len(temp) < len(Exp):
        temp = temp.reindex(ind).fillna(0)

    return ((temp - Exp) ** 2).mean()
