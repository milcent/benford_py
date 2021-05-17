from pandas import Series

from .constants import DIGS_FUNCS
from . import utils as ut

class _Ingestor_(Series):
    # def __init__(self, data, decimals, sign):
    #     Series.__init__(self, data)
    pass

def get_digits(arr, dig, decimals, sign):
    
    arr = ut._set_sign__np(arr, sign=sign)

    arr = ut._get_times_10_power_np_(arr, decimals=decimals)

    return getattr(ut, f"_get_{DIGS_FUNCS[dig]}")(arr)

class DigitsTest:
    pass

class SecondOrderDigitsTest:
    pass

class SummationTest:
    pass

class MantissasTest:
    pass
