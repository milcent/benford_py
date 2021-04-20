from contextlib import suppress as do_not_raise
import numpy as np
import pandas as pd
import pytest
from pytest import raises
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
        with pytest.raises(ValueError) as context:
            ch._check_test_(2.0)
            ch._check_test_(-3)

    def test_bool(self):
        with pytest.raises(ValueError) as context:
            ch._check_test_(False)
        with pytest.raises(ValueError) as context:
            ch._check_test_(None)


class Test_check_decimals():
    
    pos_int = []
    
    @pytest.mark.parametrize("pos_int, expected", [
        (2, 2), (0, 0), (8, 8)])
    def test_positive_int(self, pos_int, expected):
        assert ch._check_decimals_(pos_int) == expected


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

    all_confidences = [
        (None, None), (80, 80), (85, 85), (90, 90), (95, 95), (99, 99),
        (99.9, 99.9), (99.99, 99.99), (99.999, 99.999), (99.9999, 99.9999),
        (99.99999, 99.99999) 
    ]

    @pytest.mark.parametrize("conf, expected", all_confidences)
    def test_all_confidences(self, conf, expected):
        assert ch._check_confidence_(conf) == expected


class Test_check_high_Z():
        
    def test_float(self):
        with pytest.raises(ValueError) as context:
            ch._check_high_Z_(5.0)
            ch._check_high_Z_(0.3)

    def test_wrong_str(self):
        with pytest.raises(ValueError) as context:
            ch._check_high_Z_('al')
            ch._check_high_Z_('poss')

    high_Zs = [
        (10, 10), ("pos", "pos"), ("all", "all")
    ]

    @pytest.mark.parametrize("z, expected", high_Zs)
    def test_high_zs(self, z, expected):
        assert ch._check_high_Z_(z) == expected


class Test_check_nunm_array():
    
    arrays = [
        ['1', '2', '3', '4', '5', '6', '7'],
        [1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 4, 5.0, 6.3, .17],
        [True, False, False, True, True, True, False, False]
    ]

    @pytest.mark.parametrize("arr", arrays)
    def test_arrays_to_float(self, arr):
        assert ch._check_num_array_(arr).dtype == float

    def test_small_arrays(self, get_small_arrays):
        arr, expected = get_small_arrays
        assert ch._check_num_array_(arr).dtype == expected
    
    def test_np_array_str(self, small_str_foo_array):
        with pytest.raises(ValueError) as context:
            ch._check_num_array_(small_str_foo_array)

    to_raise = [
        ({1, 2, 3, 4}, raises(ValueError)),
        ({'a': 1, 'b': 2, 'c': 3, 'd': 4}, raises(ValueError)),
        ([1, 2, 3, 4, 5.0, 6.3, .17], do_not_raise()),
        (['foo', 'baar', 'baz', 'jinks'], raises(ValueError)),
        ('alocdwneceo;u', raises(ValueError))
    ]

    @pytest.mark.parametrize("num_array, expectation", to_raise)
    def test_num_array_raises(self, num_array, expectation):
        with expectation:
            assert ch._check_num_array_(num_array) is not None
