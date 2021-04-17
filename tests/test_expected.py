import pytest
from ..benford import expected as ex


class TestGetExpectedDigits():
        
    def test_1(self):
        f1d = ex._get_expected_digits_(1)
        assert type(f1d) == ex.First
        assert len(f1d) == 9

    def test_2(self):
        f1d = ex._get_expected_digits_(2)
        assert type(f1d) == ex.First
        assert len(f1d) == 90

    def test_3(self):
        f1d = ex._get_expected_digits_(3)
        assert type(f1d) == ex.First
        assert len(f1d) == 900

    def test_22(self):
        f1d = ex._get_expected_digits_(22)
        assert type(f1d) == ex.Second
        assert len(f1d) == 10

    def test_minus_2(self):
        f1d = ex._get_expected_digits_(-2)
        assert type(f1d) == ex.LastTwo
        assert len(f1d) == 100


class TestGenLastTwoDigits():
        
    def test_l2d_num_False(self):
        _, lt = ex._gen_last_two_digits_()
        assert len(lt) == 100
        assert lt.dtype == '<U21'

    def test_l2d_num_True(self):
        _, lt = ex._gen_last_two_digits_(num=True)
        assert len(lt) == 100
        assert lt.dtype == 'int'
    
    def test_exp(self):
        exp, _ = ex._gen_last_two_digits_()
        assert len(exp) == 100
        assert exp.sum() > 0.999999
        assert (exp < 0).sum() == 0

class TestGenFirstDigits():

    def test_f1d(self):
        exp, digits = ex._gen_first_digits_(1)
        assert len(exp) == len(digits) == 9
        assert exp.sum() > 0.999999
        assert digits.sum() == 45
        assert (exp < 0).sum() == 0

    
    def test_f2d(self):
        exp, digits = ex._gen_first_digits_(2)
        assert len(exp) == len(digits) == 90
        assert exp.sum() > 0.999999
        assert digits.sum() == 4905
        assert (exp < 0).sum() == 0

 
    def test_f3d(self):
        exp, digits = ex._gen_first_digits_(3)
        assert len(exp) == len(digits) == 900
        assert exp.sum() > 0.999999
        assert digits.sum() == 494550
        assert (exp < 0).sum() == 0


class TestGenSecondDigits():

    def test_gen_sec_dig(self):
        exp, digits = ex._gen_second_digits_()
        assert len(exp) == len(digits) == 10
        assert exp.sum() > 0.999999
        assert digits.sum() == 45
        assert (exp < 0).sum() == 0
