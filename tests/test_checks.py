import pytest
import pandas as pd
import numpy as np
from benford import checks as ch

def test_check_digs_zero():
    with pytest.raises(ValueError) as context:
        ch._check_digs_(0)
    assert str(context.value) == "The value assigned to the parameter -digs- was 0. Value must be 1, 2 or 3."

def test_check_digs_float():
    with pytest.raises(ValueError) as context:
        ch._check_digs_(0.5)
    assert str(context.value) == "The value assigned to the parameter -digs- was 0.5. Value must be 1, 2 or 3."

def test_check_digs_other_int():
    with pytest.raises(ValueError) as context:
        ch._check_digs_(22)
    assert str (context.value) == "The value assigned to the parameter -digs- was 22. Value must be 1, 2 or 3."
 
def test_check_digs_str():
    with pytest.raises(ValueError) as context:
        ch._check_digs_('Two')
    assert str(context.value)  == "The value assigned to the parameter -digs- was Two. Value must be 1, 2 or 3."


def test_check_test_1():
    assert ch._check_test_(1) == 1

def test_check_test_2():
    assert ch._check_test_(2) == 2

def test_check_test_3():
    assert ch._check_test_(3) == 3

def test_check_test_minus_2():
    assert ch._check_test_(-2) == -2

def test_check_test_22():
    assert ch._check_test_(22) == 22

def test_check_test_not_in_digs():
    with pytest.raises(ValueError) as context:
        ch._check_test_(4)

def test_check_test_F1D():
    assert ch._check_test_('F1D') == 1

def test_check_test_F2D():
    assert ch._check_test_('F2D') == 2

def test_check_test_F3D():
    assert ch._check_test_('F3D') == 3

def test_check_test_L2D():
    assert ch._check_test_('L2D') == -2

def test_check_test_SD():
    assert ch._check_test_('SD') == 22

def test_check_test_not_in_rev_digs():
    with pytest.raises(ValueError) as context:
        ch._check_test_('F4D')

def test_check_test_float():
    with pytest.raises(TypeError) as context:
        ch._check_test_(2.0)
        ch._check_test_(-3)

def test_check_test_bool():
    with pytest.raises(ValueError) as context:
        ch._check_test_(False)
    with pytest.raises(TypeError) as context:
        ch._check_test_(None)


def test_check_decimals_positive_int():
    assert ch._check_decimals_(2) == 2
    assert ch._check_decimals_(0) == 0
    assert ch._check_decimals_(8) == 8

def test_check_decimals_negative_int():
    with pytest.raises(ValueError) as context:
        ch._check_decimals_(-2)
    assert str(context.value) == "Parameter -decimals- must be an int >= 0, or 'infer'."
    with pytest.raises(ValueError):
        ch._check_decimals_(-5)

def test_check_decimals_infer():
    assert ch._check_decimals_('infer') == 'infer'

def test_check_decimals_other_str():
    with pytest.raises(ValueError):
        ch._check_decimals_('infe')
    with pytest.raises(ValueError):
        ch._check_decimals_('Infer')

def test_check_decimals_other_type():
    with pytest.raises(ValueError):
        assert ch._check_decimals_(-2)
    with pytest.raises(ValueError) as context:
        assert ch._check_decimals_(-2)
    assert str(context.value) == "Parameter -decimals- must be an int >= 0, or 'infer'."
    with pytest.raises(ValueError):
        assert ch._check_decimals_(None)
    with pytest.raises(ValueError) as context:
        assert ch._check_decimals_([])
        
        

def test_check_confidence_not_in_conf_keys():
    with pytest.raises(ValueError) as context:
        ch._check_confidence_(93)

def test_check_confidence_str():
    with pytest.raises(ValueError) as context:
        ch._check_confidence_('95')

def test_check_confidence_None():
    assert ch._check_confidence_(None) is None

def test_check_confidence_80():
    assert ch._check_confidence_(80) == 80

def test_check_confidence_85():
    assert ch._check_confidence_(85) == 85

def test_check_confidence_90():
    assert ch._check_confidence_(90) == 90

def test_check_confidence_95():
    assert ch._check_confidence_(95) == 95

def test_check_confidence_99():
    assert ch._check_confidence_(99) == 99

def test_check_confidence_9999():
    assert ch._check_confidence_(99.9) == 99.9

def test_check_confidence_99999():
    assert ch._check_confidence_(99.99) == 99.99

def test_check_confidence_999999():
    assert ch._check_confidence_(99.999) == 99.999

def test_check_confidence_9999999():
    assert ch._check_confidence_(99.9999) == 99.9999

def test_check_confidence_99999999():
    assert ch._check_confidence_(99.99999) == 99.99999


def test_check_high_Z_float():
    with pytest.raises(ValueError) as context:
        ch._check_high_Z_(5.0)
        ch._check_high_Z_(0.3)

def test_check_high_Z_wrong_str():
    with pytest.raises(ValueError) as context:
        ch._check_high_Z_('al')
        ch._check_high_Z_('poss')

def test_check_high_Z_int():
    assert ch._check_high_Z_(10) == 10

def test_check_high_Z_pos():
    assert ch._check_high_Z_('pos') == 'pos'
    
def test_check_high_Z_all():
    assert ch._check_high_Z_('all') == 'all'


def test_check_num_array_str():
    with pytest.raises(ValueError) as context:
        ch._check_num_array_('alocdwneceo;u')

def test_check_num_array_list_str():
    with pytest.raises(ValueError) as context:
        ch._check_num_array_(['foo','baar','baz','jinks'])

def test_check_num_array_list_of_str_num():
    assert ch._check_num_array_(['1','2','3','4','5','6','7']).dtype == float

def test_check_num_array_list_of_int():
    assert ch._check_num_array_([1,2,3,4,5,6,7]).dtype == float

def test_check_num_array_list_of_float():
    assert ch._check_num_array_([1,2,3,4,5.0,6.3,.17]).dtype == float

def test_check_num_array_npArray_float(small_float_array):
    assert ch._check_num_array_(small_float_array).dtype == float

def test_check_num_array_npArray_int(small_int_array):
    assert ch._check_num_array_(small_int_array).dtype == int

def test_check_num_array_npArray_str_num(small_str_dig_array):
    assert ch._check_num_array_(small_str_dig_array).dtype == float

def test_check_num_array_npArray_str(small_str_foo_array):
    with pytest.raises(ValueError) as context:
        ch._check_num_array_(small_str_foo_array)

def test_check_num_array_Series_float(small_float_series):
    assert ch._check_num_array_(small_float_series).dtype == float

def test_check_num_array_Series_int(small_int_series):
    assert ch._check_num_array_(small_int_series).dtype == int

def test_check_num_array_Series_str_num(small_str_dig_series):
    assert ch._check_num_array_(small_str_dig_series).dtype == float

def test_check_num_array_Series_str():
    with pytest.raises(ValueError) as context:
        ch._check_num_array_(pd.Series(['foo','baar','baz','hixks']))
    
def test_check_num_array_npArray_str_num(small_str_dig_array):
    assert ch._check_num_array_(small_str_dig_array).dtype == float

def test_check_num_array_dict():
    with pytest.raises(ValueError) as context:
        ch._check_num_array_({'a':1,'b':2,'c':3,'d':4})

def test_check_num_array_tuple():
    with pytest.raises(ValueError) as context:
        ch._check_num_array_({1,2,3,4})


