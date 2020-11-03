from pandas import Series
from numpy import array, ndarray
from .constants import digs_dict, rev_digs, confs


def _check_digs_(digs):
    """Checks the possible values for the digs parameter of the
    First Digits tests
    """
    if digs not in [1, 2, 3]:
        raise ValueError("The value assigned to the parameter -digs- "
                         f"was {digs}. Value must be 1, 2 or 3.")


def _check_test_(test):
    """Checks the test chosen, both for int or str values
    """
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


def _check_decimals_(decimals):
    """"""
    if isinstance(decimals, int):
        if (decimals < 0):
            raise ValueError(
                "Parameter -decimals- must be an int >= 0, or 'infer'.")
    else:
        if decimals != 'infer':
            raise ValueError(
                "Parameter -decimals- must be an int >= 0, or 'infer'.")
    return decimals


def _check_sign_(sign):
    """"""
    if sign not in ['all', 'pos', 'neg']:
        raise ValueError("Parameter -sign- must be one of the following: "
                         "'all', 'pos' or 'neg'.")
    return sign


def _check_confidence_(confidence):
    """"""
    if confidence not in confs.keys():
        raise ValueError("Value of parameter -confidence- must be one of the "
                         f"following:\n {list(confs.keys())}")
    return confidence


def _check_high_Z_(high_Z):
    """"""
    if not high_Z in ['pos', 'all']:
        if not isinstance(high_Z, int):
            raise ValueError("The parameter -high_Z- should be 'pos', "
                             "'all' or an int.")
    return high_Z


def _check_num_array_(data):
    """"""
    if (not isinstance(data, ndarray)) & (not isinstance(data, Series)):
        print('\n`data` not a numpy NDarray nor a pandas Series.'
              ' Trying to convert...')
        try:
            data = array(data)
        except:
            raise ValueError('Could not convert data. Check input.')
        print('\nConversion successful.')

        try:
            data = data.astype(float)
        except:
            raise ValueError('Could not convert data. Check input.')
    else:
        if data.dtype not in [int, float]:
            try:
                data = data.astype(float)
            except:
                raise ValueError('Could not convert data. Check input.')
    return data
