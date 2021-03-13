import pytest
from ..benford import expected as ex


class Test_expected():
        
    def test__1(self):
        f1d = ex._test_(1)
        assert type(f1d) == ex.First
        assert len(f1d) == 9

    def test__2(self):
        f1d = ex._test_(2)
        assert type(f1d) == ex.First
        assert len(f1d) == 90

    def test__3(self):
        f1d = ex._test_(3)
        assert type(f1d) == ex.First
        assert len(f1d) == 900

    def test__22(self):
        f1d = ex._test_(22)
        assert type(f1d) == ex.Second
        assert len(f1d) == 10

    def test__minus_2(self):
        f1d = ex._test_(-2)
        assert type(f1d) == ex.LastTwo
        assert len(f1d) == 100


class Test__lt_():
        
    def test_num_False(self):
        lt = ex._lt_()
        assert len(lt) == 100
        assert lt.dtype == '<U21'

    def test_num_True(self):
        lt = ex._lt_(num=True)
        assert len(lt) == 100
        assert lt.dtype == 'int'
