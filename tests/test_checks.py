import pytest
import pandas as pd
import numpy as np
from ..benford import checks as ch



class Test_check_digs():

    def test_zero(self):
        with pytest.raises(ValueError) as context:
            ch._check_digs_(0)
        assert str(
            context.value) == "The value assigned to the parameter -digs- was 0. Value must be 1, 2 or 3."

    def test_float(self):
        with pytest.raises(ValueError) as context:
            ch._check_digs_(0.5)
        assert str(
            context.value) == "The value assigned to the parameter -digs- was 0.5. Value must be 1, 2 or 3."

    def test_other_int(self):
        with pytest.raises(ValueError) as context:
            ch._check_digs_(22)
        assert str(
            context.value) == "The value assigned to the parameter -digs- was 22. Value must be 1, 2 or 3."

    def test_str(self):
        with pytest.raises(ValueError) as context:
            ch._check_digs_('Two')
        assert str(
            context.value) == "The value assigned to the parameter -digs- was Two. Value must be 1, 2 or 3."


class Test_check_test():
        
    def test_1(self):
        assert ch._check_test_(1) == 1

    def test_2(self):
        assert ch._check_test_(2) == 2

    def test_3(self):
        assert ch._check_test_(3) == 3

    def test_minus_2(self):
        assert ch._check_test_(-2) == -2

    def test_22(self):
        assert ch._check_test_(22) == 22

    def test_not_in_digs(self):
        with pytest.raises(ValueError) as context:
            ch._check_test_(4)

    def test_F1D(self):
        assert ch._check_test_('F1D') == 1

    def test_F2D(self):
        assert ch._check_test_('F2D') == 2

    def test_F3D(self):
        assert ch._check_test_('F3D') == 3

    def test_L2D(self):
        assert ch._check_test_('L2D') == -2

    def test_check_test_SD(self):
        assert ch._check_test_('SD') == 22

    def test_not_in_rev_digs(self):
        with pytest.raises(ValueError) as context:
            ch._check_test_('F4D')

    def test_float(self):
        with pytest.raises(TypeError) as context:
            ch._check_test_(2.0)
            ch._check_test_(-3)

    def test_bool(self):
        with pytest.raises(ValueError) as context:
            ch._check_test_(False)
        with pytest.raises(TypeError) as context:
            ch._check_test_(None)


class Test_check_decimals():
        
    def test_positive_int(self):
        assert ch._check_decimals_(2) == 2
        assert ch._check_decimals_(0) == 0
        assert ch._check_decimals_(8) == 8

    def test_negative_int(self):
        with pytest.raises(ValueError) as context:
            ch._check_decimals_(-2)
        assert str(
            context.value) == "Parameter -decimals- must be an int >= 0, or 'infer'."
        with pytest.raises(ValueError):
            ch._check_decimals_(-5)

    def test_infer(self):
        assert ch._check_decimals_('infer') == 'infer'

    def test_other_str(self):
        with pytest.raises(ValueError):
            ch._check_decimals_('infe')
        with pytest.raises(ValueError):
            ch._check_decimals_('Infer')

    def test_other_type(self):
        with pytest.raises(ValueError):
            assert ch._check_decimals_(-2)
        with pytest.raises(ValueError) as context:
            assert ch._check_decimals_(-2)
        assert str(
            context.value) == "Parameter -decimals- must be an int >= 0, or 'infer'."
        with pytest.raises(ValueError):
            assert ch._check_decimals_(None)
        with pytest.raises(ValueError) as context:
            assert ch._check_decimals_([])


class Test_check_confidence():
        
    def test_not_in_conf_keys(self):
        with pytest.raises(ValueError) as context:
            ch._check_confidence_(93)

    def test_str(self):
        with pytest.raises(ValueError) as context:
            ch._check_confidence_('95')

    def test_None(self):
        assert ch._check_confidence_(None) is None

    def test_80(self):
        assert ch._check_confidence_(80) == 80

    def test_85(self):
        assert ch._check_confidence_(85) == 85

    def test_90(self):
        assert ch._check_confidence_(90) == 90

    def test_95(self):
        assert ch._check_confidence_(95) == 95

    def test_99(self):
        assert ch._check_confidence_(99) == 99

    def test_9999(self):
        assert ch._check_confidence_(99.9) == 99.9

    def test_99999(self):
        assert ch._check_confidence_(99.99) == 99.99

    def test_999999(self):
        assert ch._check_confidence_(99.999) == 99.999

    def test_9999999(self):
        assert ch._check_confidence_(99.9999) == 99.9999

    def test_99999999(self):
        assert ch._check_confidence_(99.99999) == 99.99999


class Test_check_high_Z():
        
    def test_float(self):
        with pytest.raises(ValueError) as context:
            ch._check_high_Z_(5.0)
            ch._check_high_Z_(0.3)

    def test_wrong_str(self):
        with pytest.raises(ValueError) as context:
            ch._check_high_Z_('al')
            ch._check_high_Z_('poss')

    def test_int(self):
        assert ch._check_high_Z_(10) == 10

    def test_pos(self):
        assert ch._check_high_Z_('pos') == 'pos'

    def test_all(self):
        assert ch._check_high_Z_('all') == 'all'


class Test_check_nunm_array():
        
    def test_str(self):
        with pytest.raises(ValueError) as context:
            ch._check_num_array_('alocdwneceo;u')

    def test_list_str(self):
        with pytest.raises(ValueError) as context:
            ch._check_num_array_(['foo', 'baar', 'baz', 'jinks'])

    def test_list_of_str_num(self):
        assert ch._check_num_array_(
            ['1', '2', '3', '4', '5', '6', '7']).dtype == float

    def test_list_of_int(self):
        assert ch._check_num_array_([1, 2, 3, 4, 5, 6, 7]).dtype == float

    def test_list_of_float(self):
        assert ch._check_num_array_([1, 2, 3, 4, 5.0, 6.3, .17]).dtype == float

    def test_npArray_float(self, small_float_array):
        assert ch._check_num_array_(small_float_array).dtype == float

    def test_npArray_int(self, small_int_array):
        assert ch._check_num_array_(small_int_array).dtype == int

    def test_npArray_str_num(self, small_str_dig_array):
        assert ch._check_num_array_(small_str_dig_array).dtype == float

    def test_npArray_str(self, small_str_foo_array):
        with pytest.raises(ValueError) as context:
            ch._check_num_array_(small_str_foo_array)

    def test_Series_float(self, small_float_series):
        assert ch._check_num_array_(small_float_series).dtype == float

    def test_Series_int(self, small_int_series):
        assert ch._check_num_array_(small_int_series).dtype == int

    def test_Series_str_num(self, small_str_dig_series):
        assert ch._check_num_array_(small_str_dig_series).dtype == float

    def test_Series_str(self):
        with pytest.raises(ValueError) as context:
            ch._check_num_array_(pd.Series(['foo', 'baar', 'baz', 'hixks']))

    def test_npArray_str_num(self, small_str_dig_array):
        assert ch._check_num_array_(small_str_dig_array).dtype == float

    def test_dict(self):
        with pytest.raises(ValueError) as context:
            ch._check_num_array_({'a': 1, 'b': 2, 'c': 3, 'd': 4})

    def test_tuple(self):
        with pytest.raises(ValueError) as context:
            ch._check_num_array_({1, 2, 3, 4})
