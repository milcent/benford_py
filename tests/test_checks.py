from contextlib import suppress as do_not_raise
import pytest
from pytest import raises
from ..benford import checks as ch
from ..benford.constants import confs, digs_dict


class TestCheckDigs():

    digs_to_raise = [
        (x, raises(ValueError)) for x in 
        [0, 0.5, -3, -5, -1, 1.7, 22, 1000, "One", "Two", "Second", "LastTwo", "Three"]
    ]

    @pytest.mark.parametrize("dig, expectation", digs_to_raise)
    def test_digs_raise_msg(self, dig, expectation):
        with expectation as context:
            ch._check_digs_(dig)
        assert str(context.value) == "The value assigned to the parameter " +\
                                f"-digs- was {dig}. Value must be 1, 2 or 3."

    @pytest.mark.parametrize("dig, expectation", digs_to_raise)
    def test_check_digs_raise(self, dig, expectation):
        with expectation:
            assert ch._check_digs_(dig) is not None

    legit_digs = [
        (y, do_not_raise()) for y in [1, 2, 3]
    ]
    @pytest.mark.parametrize("dig, expectation", legit_digs)
    def test_check_digs_no_raise(self, dig, expectation):
        with expectation:
            assert ch._check_digs_(dig) is None


class TestCheckTest():

    digs_tests = [(d, d) for d in digs_dict.keys()] +\
        [(val, key) for key, val in digs_dict.items()]
    
    @pytest.mark.parametrize("dig, expected", digs_tests)
    def test_choose(self, dig, expected):
        assert ch._check_test_(dig) == expected   
    
    test_check_raise = [
        (y, raises(ValueError)) for y in [4, -3, 2.0, "F4D", False]] +\
        [(x, do_not_raise()) for x in digs_dict.keys()] +\
        [(z, do_not_raise()) for z in digs_dict.values()]

    @pytest.mark.parametrize("dig, expectation", test_check_raise)
    def test_raise(self, dig, expectation):
        with expectation:
            assert ch._check_test_(dig) is not None

    def test_None(self):
        with pytest.raises(ValueError):
            ch._check_test_(None)


class TestCheckDecimals():
    
    pos_int = zip(range(21), range(21))
    
    @pytest.mark.parametrize("pos_int, expected", pos_int)
    def test_positive_int(self, pos_int, expected):
        assert ch._check_decimals_(pos_int) == expected

    dec_errors = [(x, raises(ValueError)) for x in range(-15, 0)] +\
        [(y, do_not_raise()) for y in range(21)] +\
        [(z, raises(ValueError)) for z in ["inf", "infe", "Infer", []]]

    @pytest.mark.parametrize("dec, expectation", dec_errors)
    def test_dec_raises(self, dec, expectation):
        with expectation:
            assert ch._check_decimals_(dec) is not None

    def test_negative_int_msg(self):
        with pytest.raises(ValueError) as context:
            ch._check_decimals_(-2)
        assert str(
            context.value) == "Parameter -decimals- must be an int >= 0, or 'infer'."

    def test_infer(self):
        assert ch._check_decimals_('infer') == 'infer'

    def test_None_type(self):
        with pytest.raises(ValueError):
            ch._check_decimals_(None)


class TestCheckConfidence():

    conf_errors = [
        (x, raises(ValueError)) for x in
        [93, "95", 76, "80", "99", 84, 99.8]
    ] + [ # Except None ([:1]) due to comparison below
        (y, do_not_raise()) for y in list(confs.keys())[1:] 
    ]
    @pytest.mark.parametrize("conf, expectation", conf_errors)
    def test_conf_raises(self, conf, expectation):
        with expectation:
            assert ch._check_confidence_(conf) is not None

    all_confidences = zip(confs.keys(), confs.keys())

    @pytest.mark.parametrize("conf, expected", all_confidences)
    def test_all_confidences(self, conf, expected):
        assert ch._check_confidence_(conf) == expected


class TestCheckHighZ():

    z_errors = [
        (x, raises(ValueError)) for x in
        [5.0, 0.3, "al", "poss", "po", "alll", ]
    ] + [
        (y, do_not_raise()) for y in 
        [10, 20, 5, 2, "pos", "all"]
    ]
    @pytest.mark.parametrize("high_Z, expectation", z_errors)
    def test_high_Z_raises(self, high_Z, expectation):
        with expectation:
            assert ch._check_high_Z_(high_Z) is not None

    high_Zs = [
        (10, 10), ("pos", "pos"), ("all", "all")
    ]
    @pytest.mark.parametrize("z, expected", high_Zs)
    def test_high_zs(self, z, expected):
        assert ch._check_high_Z_(z) == expected


class TestCheckNunmArray():
    
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
        with pytest.raises(ValueError):
            ch._check_num_array_(small_str_foo_array)

    num_arr_raise = [
        ({1, 2, 3, 4}, raises(ValueError)),
        ({'a': 1, 'b': 2, 'c': 3, 'd': 4}, raises(ValueError)),
        ([1, 2, 3, 4, 5.0, 6.3, .17], do_not_raise()),
        (['foo', 'baar', 'baz', 'jinks'], raises(ValueError)),
        ('alocdwneceo;u', raises(ValueError))
    ]

    @pytest.mark.parametrize("num_array, expectation", num_arr_raise)
    def test_num_array_raises(self, num_array, expectation):
        with expectation:
            print(num_array)
            assert ch._check_num_array_(num_array) is not None
