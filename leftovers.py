import numopy as np
import pandas as pd


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


def _str_to_float_(s, dec=2):
    if '.' in s or ',' in s:
        s = list(filter(type(s).isdigit, s))
        s = s[:-dec] + '.' + s[-dec:]
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
    li = 10 ** int(np.log10(num))
    # If dec is different than 0, use dec to round to the decimal
    # places chosen
    if dec != 0:
        return round(10. ** (orders + 1 - n) * num / li, dec)
    # If dec == 0, return integer
    else:
        return int(10. ** (orders + 1 - n) * num / li)


def _collapse_array_(arr, orders=2, dec=2):
    '''
    Collapses an array of numbers, each to a form defined by the user,
    with the chosen number of digits at the left of the floating point,
    with the chosen number of decimal digits or as ints.

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
    n[arr >= 1] = 2
    # Set the dividend array l, composed of numbers 10 raised to the
    # integer part of the numbers' logs
    li = 10. ** (np.log10(arr).astype(int, copy=False))
    # If dec is different than 0, use dec to round to the decimal
    # places chosen
    if dec != 0:
        return 10. ** (orders + 1 - n) * arr / li
    # If dec == 0, return array of integers
    else:
        return (10. ** (orders + 1 - n) * arr / li).astype(int)


def _sanitize_float_(s, dec):
    s = str(s)
    if '.' in s or ',' in s:
        s = list(filter(type(s).isdigit, s))
        s = s[:-dec] + '.' + s[-dec:]
        return float(s)
    else:
        if list(filter(type(s).isdigit, s)) == '':
            return np.nan
        else:
            return int(s)


def _sanitize_latin_float_(s, dec=2):
    s = str(s)
    s = list(filter(type(s).isdigit, s))
    return s[:-dec] + '.' + s[-dec:]


def _sanitize_latin_int_(s):
    s = str(s)
    s = list(filter(type(s).isdigit, s))
    return s
