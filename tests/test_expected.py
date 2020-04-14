import pytest
from benford import expected as ex

def test__test_1():
    f1d = ex._test_(1)
    assert type(f1d) == ex.First
    assert len(f1d) == 9

def test__test_2():
    f1d = ex._test_(2)
    assert type(f1d) == ex.First
    assert len(f1d) == 90

def test__test_3():
    f1d = ex._test_(3)
    assert type(f1d) == ex.First
    assert len(f1d) == 900

def test__test_22():
    f1d = ex._test_(22)
    assert type(f1d) == ex.Second
    assert len(f1d) == 10

def test__test_minus_2():
    f1d = ex._test_(-2)
    assert type(f1d) == ex.LastTwo
    assert len(f1d) == 100


def test__lt_num_False():
    lt = ex._lt_()
    assert len(lt) == 100
    assert lt.dtype == '<U21'

def test__lt_num_True():
    lt = ex._lt_(num=True)
    assert len(lt) == 100
    assert lt.dtype == 'int64'

