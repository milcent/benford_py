import pytest
from ..benford import expected as ex


class TestGetExpectedDigits():

    expected_types = [
        (x, ex.First) for x in [1, 2, 3]
    ] + [(22, ex.Second), (-2, ex.LastTwo)]

    @pytest.mark.parametrize("dig, expec_type", expected_types)
    def test_expected_types(self, dig, expec_type):
        assert type(ex._get_expected_digits_(dig)) == expec_type

    expected_lenghts = [
        (1, 9), (2, 90), (3, 900), (22, 10), (-2, 100)
    ]

    @pytest.mark.parametrize("dig, exp_len", expected_lenghts)
    def test_expected_lenghts(self, dig, exp_len):
        assert len(ex._get_expected_digits_(dig)) == exp_len


class TestGenLastTwoDigits():

    l2d_types = [([], "<U21"), ([True], "int")]

    @pytest.mark.parametrize("num, arr_type", l2d_types)
    def test_types(self, num, arr_type):
        _, lt = ex._gen_last_two_digits_(*num)
        assert lt.dtype == arr_type


class TestGenDigits():

    gen_digs = [
        ("_gen_first_digits_", [1]), ("_gen_first_digits_", [2]),
        ("_gen_first_digits_", [3]), ("_gen_second_digits_", []),
        ("_gen_last_two_digits_", []), ("_gen_last_two_digits_", [True])
    ]

    @pytest.mark.parametrize("func, dig", gen_digs)
    def test_probs_sum_near_one(self, func, dig):
        exp, _ = getattr(ex, func)(*dig)
        assert exp.sum() > 0.999999

    @pytest.mark.parametrize("func, dig", gen_digs)
    def test_no_negative_prob(self, func, dig):
        exp, _ = getattr(ex, func)(*dig)
        assert (exp < 0).sum() == 0

    digs_sums = [
        ("_gen_first_digits_", [1], 45), ("_gen_first_digits_", [2], 4905),
        ("_gen_first_digits_", [3], 494550), ("_gen_second_digits_",  [], 45),
        ("_gen_last_two_digits_", [True], 4950) 
    ]

    @pytest.mark.parametrize("func, dig, exp_sum", digs_sums)
    def test_digs_sums(self, func, dig, exp_sum):
        _, digits = getattr(ex, func)(*dig)
        assert digits.sum() == exp_sum

    digs_lengths = [
        ("_gen_first_digits_", [1], 9), ("_gen_first_digits_", [2], 90),
        ("_gen_first_digits_", [3], 900), ("_gen_second_digits_",  [], 10),
        ("_gen_last_two_digits_", [], 100), ("_gen_last_two_digits_", [True], 100) 
    ]

    @pytest.mark.parametrize("func, dig, exp_len", digs_lengths)
    def test_lengths(self, func, dig, exp_len):
        exp, digits = getattr(ex, func)(*dig)
        assert len(exp) == len(digits) == exp_len
